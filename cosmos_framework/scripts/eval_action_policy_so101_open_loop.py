# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Open-loop evaluation of an SO101 action policy (``action_policy_so101_edge``) on recorded trajectories.

Each trajectory is walked in strides of ``horizon`` frames for up to ``steps`` frames. At every stride the policy gets
the recorded frame (top + wrist, composed as in training) and state, and its first ``horizon`` actions are compared
with the recorded ones in the dataset's own units. This is the protocol of the other SO101 open-loop evaluations
(openpi ``eval_so101_open_loop.py``), so the numbers are comparable.

Usage (1 GPU)::

    PYTHONPATH=. python -m cosmos_framework.scripts.eval_action_policy_so101_open_loop \\
        --checkpoint-path $RUN_DIR/checkpoints/iter_000006000 --root $SO101_ROOT \\
        --traj-ids 0 25 51 80 108 133 --norm-std 29.97 --output-dir eval/so101_edge
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from cosmos_framework.data.generator.action.datasets.so101_lerobot_dataset import (
    _CONCAT_VIEW_DESCRIPTION,
    _TOP_FEATURE,
    _WRIST_FEATURE,
    SO101LeRobotDataset,
    so101_unscale_action,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec


@dataclass
class Args:
    checkpoint_path: str
    """Trained DCP checkpoint dir, e.g. $RUN_DIR/checkpoints/iter_000006000."""
    root: str
    """Local SO101 LeRobot v3.0 dir."""
    output_dir: str
    traj_ids: list[int] = field(default_factory=lambda: [0, 25, 51, 80, 108, 133])
    steps: int = 400
    horizon: int = 16
    norm_std: float | None = None
    """Absolute action std; when set, MAE is also reported as a percentage of it."""
    experiment: str = "action_policy_so101_edge"
    vae_path: str = "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
    num_steps: int = 30
    guidance: float = 1.0
    shift: float = 5.0
    seed: int = 0
    image_size: int = 256


class SO101PolicyService(RobolabPolicyService):
    """The RoboLab policy service with the SO101 inputs: concat_view top + wrist and the 6D state row."""

    def __init__(self, args: RobolabServerArgs, image_size: int) -> None:
        super().__init__(args)
        self._image_size = image_size

    def build_so101_sample(self, prompt: str, top: torch.Tensor, wrist: torch.Tensor, state: np.ndarray) -> dict:
        """``top``/``wrist`` are [3, H, W] in [0, 1]; ``state`` is the scaled 6D state."""
        size = (self._image_size, self._image_size)
        views = [
            F.interpolate(v.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0] for v in (top, wrist)
        ]
        frame = (torch.cat(views, dim=-1) * 255.0).clamp(0.0, 255.0).to(torch.uint8)  # [3, H, 2W]
        t_frames = self.cfg.action_chunk_size + 1
        video = torch.zeros((3, t_frames, *frame.shape[1:]), dtype=torch.uint8)
        video[:, 0] = frame
        action = torch.zeros((self.cfg.action_chunk_size + 1, 6), dtype=torch.float32)
        action[0] = torch.from_numpy(state)
        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.cfg.conditioning_fps, dtype=torch.long),
            "mode": "wam",
            "domain_id": torch.tensor(get_domain_id(self.cfg.domain_name), dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        }
        sample = self._transform(sample, self.cfg.resolution)
        if isinstance(sample.get("ai_caption"), dict):
            sample["ai_caption"] = json.dumps(sample["ai_caption"])
        return sample

    def predict(self, sample: dict) -> np.ndarray:
        """Scaled [chunk, 6] actions, without the state row."""
        samples = self._generate(sample, self._next_seed())
        return samples["action"][0][1:, :6].float().cpu().numpy()


def main(args: Args) -> None:
    dataset = SO101LeRobotDataset(root=args.root, chunk_length=args.horizon, split="full", image_size=args.image_size)
    service = SO101PolicyService(
        RobolabServerArgs(
            checkpoint_path=args.checkpoint_path,
            allow_dcp_checkpoint=True,
            experiment=args.experiment,
            experiment_overrides=[f"model.config.tokenizer.vae_path={args.vae_path}"],
            domain_name="so101_joint",
            seed=args.seed,
            guidance=args.guidance,
            num_steps=args.num_steps,
            shift=args.shift,
            resolution=None,
            conditioning_fps=dataset.fps,
            action_chunk_size=args.horizon,
            action_dim=6,
            use_state=True,
            history_length=1,
            format_prompt_as_json=True,
        ),
        image_size=args.image_size,
    )
    load_torchcodec()
    from lerobot.datasets.video_utils import decode_video_frames

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for tid in args.traj_ids:
        ep = int(np.flatnonzero(dataset._ep_vals == tid)[0])
        start = int(dataset._ep_starts[ep])
        length = int(np.sum(dataset._row_episode == tid))
        episode = dataset._episodes[tid]
        prompt = dataset._tasks[int(dataset._row_task[start])]
        n = min(args.steps, length - args.horizon)
        preds, gts = [], []
        for t in range(0, n, args.horizon):
            ts = float(dataset._row_timestamp[start + t])
            frames = [
                decode_video_frames(
                    dataset._video_path(episode, key),
                    [float(episode.get(f"videos/{key}/from_timestamp", 0.0)) + ts],
                    dataset._tolerance_s,
                )[0]
                for key in (_TOP_FEATURE, _WRIST_FEATURE)
            ]
            sample = service.build_so101_sample(prompt, frames[0], frames[1], dataset._row_state[start + t])
            preds.append(so101_unscale_action(service.predict(sample)[: args.horizon]))
            gts.append(so101_unscale_action(dataset._row_action[start + t : start + t + args.horizon]))
        pred, gt = np.concatenate(preds), np.concatenate(gts)
        err = pred - gt
        results[str(tid)] = {"steps": len(err), "mse": float((err**2).mean()), "mae": float(np.abs(err).mean())}
        np.savez(out / f"traj_{tid}.npz", pred=pred, gt=gt)
        log.info(
            f"traj {tid}: MSE {results[str(tid)]['mse']:.4f} MAE {results[str(tid)]['mae']:.4f} ({len(err)} steps)"
        )
    mean = {k: float(np.mean([r[k] for r in results.values()])) for k in ("mse", "mae")}
    if args.norm_std:
        mean["mae_pct"] = mean["mae"] / args.norm_std * 100
    summary = {
        "checkpoint": args.checkpoint_path,
        "horizon": args.horizon,
        "steps": args.steps,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "trajectories": results,
        "mean": mean,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    log.info(f"mean: {mean}")


if __name__ == "__main__":
    main(tyro.cli(Args))
