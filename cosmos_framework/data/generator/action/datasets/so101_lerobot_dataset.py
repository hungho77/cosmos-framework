# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""SO101 single-arm LeRobot dataset (absolute joint-position action policy).

Mirrors ``LIBEROLeRobotDataset``: reads the LeRobot v3.0 parquet directly, windows by
frame index, and decodes video at each frame's real timestamp.

Action layout (``joint_pos``), as in the DROID ``joint_pos`` recipe: the stored 6D
``action`` is the absolute target ``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper]``. The five arm joints are scaled by pi/180 and the gripper by 1/100,
so the arm channels have the magnitude of DROID's radians and the gripper lies in
[0, 1]; no further normalization is applied. With ``use_state`` the state at the
conditioning frame is prepended as one clean row, ``[chunk + 1, 6]``.

Video: ``concat_view`` puts the overhead ``top`` camera on the left and the ``wrist``
camera on the right, each resized to ``image_size`` x ``image_size``. Other video
features (e.g. ``observation.images.top_depth``) are ignored.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Joint, build_action_spec
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

CameraMode = Literal["top", "wrist", "concat_view"]

_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"
_TOP_FEATURE = "observation.images.top"
_WRIST_FEATURE = "observation.images.wrist"
# Arm joints are stored in degrees-like units (about +-100), the gripper in [0, 100].
_SCALE = np.array([np.pi / 180.0] * 5 + [1.0 / 100.0], dtype=np.float32)

_VIEWPOINT_BY_CAMERA = {
    "top": "third_person_view",
    "wrist": "wrist_view",
    "concat_view": "concat_view",
}
_CONCAT_VIEW_DESCRIPTION = "The left half shows the overhead camera; the right half shows the wrist-mounted camera."


def so101_scale_action(action: np.ndarray) -> np.ndarray:
    """Dataset units -> model units (arm joints x pi/180, gripper / 100)."""
    return action * _SCALE


def so101_unscale_action(action: np.ndarray) -> np.ndarray:
    """Model units -> dataset units."""
    return action / _SCALE


class SO101LeRobotDataset(ActionBaseDataset):
    """SO101 action-policy dataset: 6D absolute joint positions, optional state row,
    ``concat_view`` top + wrist video."""

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 16,
        mode: str = "wam",
        tolerance_s: float = 1e-4,
        camera_mode: CameraMode = "concat_view",
        image_size: int = 256,
        use_state: bool = True,
        embodiment_type: str = "so101_joint",
        split: str = "train",
        val_ratio: float = 0.01,
        seed: int = 0,
        sample_stride: int = 1,
    ) -> None:
        if camera_mode not in _VIEWPOINT_BY_CAMERA:
            raise ValueError(f"Unsupported camera_mode={camera_mode!r}. Use top/wrist/concat_view.")
        split = split.lower().strip()
        if split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported split={split!r}. Use train/val/full.")
        if chunk_length % 4 != 0:
            raise ValueError(f"chunk_length must be divisible by 4, got {chunk_length}.")

        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",  # unused for joint positions; satisfies the base assert
            tolerance_s=tolerance_s,
            viewpoint=_VIEWPOINT_BY_CAMERA[camera_mode],
            action_normalization=None,  # scaled joint positions are used as is, as in the DROID joint_pos recipe
            sample_stride=sample_stride,
        )
        info_fps = self._info.get("fps")
        if info_fps and int(info_fps) != int(fps):
            log.info(f"Using dataset native fps={info_fps} for conditioning (requested {fps}).")
            self._fps = float(info_fps)
            self._dt = 1.0 / self._fps
        self._camera_mode = camera_mode
        self._image_size = int(image_size)
        self._use_state = use_state
        if camera_mode == "top":
            self._video_keys = [_TOP_FEATURE]
        elif camera_mode == "wrist":
            self._video_keys = [_WRIST_FEATURE]
        else:
            self._video_keys = [_TOP_FEATURE, _WRIST_FEATURE]

        # Compact frame index ordered by global frame index (see LIBEROLeRobotDataset).
        index_parts, episode_parts, task_parts, ts_parts, action_parts, state_parts = [], [], [], [], [], []
        columns = ["index", "episode_index", "task_index", "timestamp", _ACTION_FEATURE, _STATE_FEATURE]
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=columns)
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            action_parts.append(np.asarray(table[_ACTION_FEATURE].to_pylist(), dtype=np.float32))
            state_parts.append(np.asarray(table[_STATE_FEATURE].to_pylist(), dtype=np.float32))
        if not index_parts:
            raise FileNotFoundError(f"No data parquet found under {self._root / 'data'}.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_action = so101_scale_action(np.concatenate(action_parts, axis=0)[order])
        self._row_state = so101_scale_action(np.concatenate(state_parts, axis=0)[order])

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)

        keep = self._split_episode_ids(ep_vals.tolist(), split, val_ratio, seed)
        kept = np.array([int(v) in keep for v in ep_vals], dtype=bool)
        self._ep_vals = ep_vals.astype(np.int64)[kept]
        self._ep_starts = ep_starts.astype(np.int64)[kept]
        kept_counts = ep_counts.astype(np.int64)[kept]
        # A window needs chunk_length + 1 frames (the conditioning frame and the chunk).
        self._valid_cum = np.cumsum(np.maximum(0, kept_counts - self._chunk_length)).astype(np.int64)

        log.info(
            f"Loaded SO101 dataset root={self._root} split={split!r} camera_mode={camera_mode!r} "
            f"fps={self._fps} use_state={use_state} kept_episodes={len(self._ep_vals)}/{len(ep_vals)} "
            f"valid_indices={int(self._valid_cum[-1]) if self._valid_cum.size else 0}"
        )

    # ---- spec / dims -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 6

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Joint(n=5, label="joint"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        raise NotImplementedError("SO101LeRobotDataset uses no action normalization stats.")

    # ---- index helpers -----------------------------------------------------

    @staticmethod
    def _split_episode_ids(ep_ids: list[int], split: str, val_ratio: float, seed: int) -> set[int]:
        if split == "full":
            return set(int(v) for v in ep_ids)
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
        n_val = max(1, int(round(len(ep_ids) * val_ratio)))
        rng = random.Random(seed)  # identical selection on every rank
        val = set(int(v) for v in rng.sample(list(ep_ids), n_val))
        if split == "train":
            return set(int(v) for v in ep_ids) - val
        return val

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(start, length)`` flat-index blocks for ``ActionIterableShuffleDataset``."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    # ---- sample build ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self)
        last_err: Exception | None = None
        for _attempt in range(8):
            try:
                return self._build_item(idx)
            except Exception as e:  # noqa: BLE001 — skip past undecodable frames
                last_err = e
                log.warning(f"SO101: sample idx={idx} failed to load ({type(e).__name__}: {e}); resampling")
                if n > 0:
                    idx = random.randint(0, n - 1)
        raise RuntimeError(f"SO101: failed to load a sample after 8 resamples; last error: {last_err}")

    def _build_item(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode = self._episodes[int(self._ep_vals[ep])]

        stop = start + self._chunk_length + 1
        timestamps = [float(self._row_timestamp[j]) for j in range(start, stop)]
        video = self._load_video(episode, timestamps)

        # The action at frame t is the target that leads to frame t + 1.
        action = torch.from_numpy(np.ascontiguousarray(self._row_action[start : start + self._chunk_length]))
        if self._use_state:
            state = torch.from_numpy(np.ascontiguousarray(self._row_state[start]))
            action = torch.cat([state.unsqueeze(0), action], dim=0)  # [chunk + 1, 6]

        task = self._tasks[int(self._row_task[start])]
        extras: dict[str, Any] = {}
        if self._camera_mode == "concat_view":
            extras["additional_view_description"] = _CONCAT_VIEW_DESCRIPTION
        return self._build_result(mode=mode, video=video, action=action.float(), ai_caption=task, **extras)

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        # Preload the wheel-installed NPP before LeRobot imports TorchCodec, also in DataLoader workers.
        load_torchcodec()
        from lerobot.datasets.video_utils import decode_video_frames

        frames_by_view = {}
        for key in self._video_keys:
            from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            frames = decode_video_frames(
                self._video_path(episode, key),
                [from_ts + ts for ts in timestamps],
                self._tolerance_s,
            )  # [T, C, H, W] in [0, 1]
            frames_by_view[key] = self._resize(frames)
        if self._camera_mode == "concat_view":
            return torch.cat([frames_by_view[_TOP_FEATURE], frames_by_view[_WRIST_FEATURE]], dim=-1)  # [T, C, H, 2W]
        return frames_by_view[self._video_keys[0]]

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] == self._image_size and frames.shape[-2] == self._image_size:
            return frames
        return F.interpolate(frames, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)
