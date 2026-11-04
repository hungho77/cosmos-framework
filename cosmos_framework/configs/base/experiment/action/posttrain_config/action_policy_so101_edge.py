# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_so101_edge`` — Cosmos3-Edge SO101 action-policy SFT recipe.

The LIBERO recipe (``action_policy_libero_nano``: optimizer, schedule, EMA, trained
keys, 16-action chunks, concat_view 256x256 views) on the Cosmos3-Edge model, fed by
``SO101LeRobotDataset``: 6D absolute joint positions with the state row, as in the
DROID ``joint_pos`` recipe. ``SO101_ROOT`` is a local LeRobot v3.0 dir with the
``observation.images.top`` and ``observation.images.wrist`` cameras.

Usage (1 GPU)::

    SO101_ROOT=<dir> BASE_CHECKPOINT_PATH=<Cosmos3-Edge DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
        bash examples/launch_sft_action_policy_so101_edge.sh
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
    action_policy_libero_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_so101_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


def _action_policy_so101_edge_model_config() -> dict:
    """The LIBERO model tweaks applied to the Edge baseline (Edge already has the 10x vision loss)."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)  # action_gen=True, max_action_dim=64
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]  # chunk 16 -> 17 frames, as for LIBERO
    return cfg


action_policy_so101_edge = copy.deepcopy(action_policy_libero_nano)
action_policy_so101_edge["job"]["name"] = "action_policy_so101_edge"
action_policy_so101_edge["model"]["config"] = _action_policy_so101_edge_model_config()
action_policy_so101_edge["dataloader_train"]["dataset_name"] = "action_so101"
# A sample takes ~0.2 s of CPU to decode (two AV1 views x 17 frames); use most of an 8-core host.
action_policy_so101_edge["dataloader_train"]["dataloader"]["num_workers"] = 6
action_policy_so101_edge["dataloader_train"]["dataloader"]["datasets"] = dict(
    so101=dict(
        ratio=1,
        dataset=L(get_action_so101_sft_dataset)(
            root="${oc.env:SO101_ROOT}",
            fps=30,
            chunk_length=16,
            image_size=256,  # concat_view -> 256x512
            mode="wam",
            camera_mode="concat_view",
            use_state=True,
            split="full",  # every episode trains, as in the other SO101 baselines (pi0.5, GR00T)
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            resolution=None,
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            format_prompt_as_json=True,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
        ),
    ),
)


for _item in [action_policy_so101_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
