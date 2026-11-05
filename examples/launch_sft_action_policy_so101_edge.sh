#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_so101_edge — Cosmos3-Edge SO101 action-policy
# SFT on one GPU. Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_so101_edge.toml.
#
# Required env vars:
#   SO101_ROOT            local SO101 LeRobot v3.0 dir with the top and wrist cameras (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   OUTPUT_ROOT           default: outputs/train
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides, e.g. "trainer.max_iter=50" for a pilot
#
# Usage:
#   SO101_ROOT=<dir> bash examples/launch_sft_action_policy_so101_edge.sh

TOML_FILE="examples/toml/sft_config/action_policy_so101_edge.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${NPROC_PER_NODE:=1}"

export SO101_ROOT="${SO101_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$SO101_ROOT/meta/info.json" ]] || { echo "ERROR: SO101_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$SO101_ROOT'\'')." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
