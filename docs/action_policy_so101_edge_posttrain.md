# Cosmos3-Edge SO101 Action-Policy Post-Training

This recipe post-trains [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) into an action policy for a single
SO101 arm, on one GPU. Experiment `action_policy_so101_edge`, launcher `launch_sft_action_policy_so101_edge.sh`.

| Piece        | Path                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| Dataset      | `cosmos_framework/data/generator/action/datasets/so101_lerobot_dataset.py` (`SO101LeRobotDataset`)   |
| SFT wrapper  | `get_action_so101_sft_dataset` in `.../datasets/action_sft_dataset.py`                               |
| Experiment   | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_so101_edge.py`       |
| Run TOML     | `examples/toml/sft_config/action_policy_so101_edge.toml`                                             |
| Launch       | `examples/launch_sft_action_policy_so101_edge.sh`                                                    |
| Open-loop eval | `cosmos_framework/scripts/eval_action_policy_so101_open_loop.py`                                   |

## Recipe

The LIBERO recipe (`action_policy_libero_nano`) on the Edge model, with the action layout of the DROID `joint_pos`
recipe:

| knob          | value                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------- |
| init          | `Cosmos3-Edge`; the action modules start fresh                                                     |
| action        | 6D absolute joint positions (5 joints + gripper); joints x pi/180, gripper / 100, no normalization |
| state         | the state at the conditioning frame is a clean first action row (`use_state=True`)                 |
| video         | `concat_view`: overhead `top` camera left, `wrist` camera right, 256x256 each (192x320 canvas)      |
| chunk         | 16 actions (17 frames), 30 FPS                                                                     |
| embodiment    | `so101_joint` (domain 25)                                                                          |
| optimizer     | FusedAdam, lr 5e-5 (5x for the action modules), warmup 500, linear decay to 0 at 6000, power EMA   |
| batch         | 32 per micro-batch x 4 accumulation steps = 128 on one GPU                                          |
| split         | every episode trains                                                                               |

## Data

`SO101_ROOT` is a local LeRobot v3.0 dir with `observation.images.top`, `observation.images.wrist`,
`observation.state` and `action` (6D each, LeRobot SO101 units). Other video features, such as
`observation.images.top_depth`, are ignored. Several datasets can be merged into one dir with
`lerobot.datasets.aggregate.aggregate_datasets`.

## Train

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp -o examples/checkpoints/Cosmos3-Edge --checkpoint-path Cosmos3-Edge
export LD_LIBRARY_PATH=''
export SO101_ROOT=<dir>
bash examples/launch_sft_action_policy_so101_edge.sh
```

On one H100 80GB this uses about 50 GB and runs at about 6 s per optimizer step (128 samples), so the 6000 steps
take about 10 h. With 8 CPU cores the AV1 video decoding of the dataloader is the bottleneck.

## Evaluate

```bash
PYTHONPATH=. python -m cosmos_framework.scripts.eval_action_policy_so101_open_loop \
  --checkpoint-path $RUN_DIR/checkpoints/iter_000006000 --root $SO101_ROOT \
  --traj-ids 0 25 51 80 108 133 --norm-std 29.97 --output-dir outputs/eval/so101_edge
```

The script walks each trajectory in 16-frame strides for up to 400 frames, queries the policy on the recorded frame
and state, and reports the MAE of the first 16 actions in the dataset's units. It loads the DCP checkpoint directly.

## Results

`hungho77/so101-left-arm-task3` (3 tasks, 159 episodes), trajectories 0 25 51 80 108 133, MAE in the dataset's units
and as a percentage of the absolute action std (29.97). The other rows are the same protocol on the same data.

| model                          | MAE   | % std |
| ------------------------------ | ----- | ----- |
| GR00T N1.7                     | 1.440 | 4.80  |
| pi0.5                          | 1.459 | 4.87  |
| this recipe, step 6000         | 1.911 | 6.38  |
| this recipe, step 4000         | 2.444 | 8.15  |

Step 6000 is better than step 4000 on every trajectory and the loss is still falling at the end, so the run is
probably undertrained. The error is largest on the elbow and wrist-flex joints and grows along the chunk (0.95 at the
first action, 2.89 at the 16th).

The sampling settings change little at step 6000 (default: 30 steps, guidance 1, shift 5, seed 0):

| setting      | MAE   | % std |
| ------------ | ----- | ----- |
| default      | 1.911 | 6.38  |
| seed 1       | 1.944 | 6.49  |
| 10 steps     | 1.908 | 6.37  |
| 50 steps     | 1.940 | 6.47  |
| guidance 1.5 | 1.942 | 6.48  |
| guidance 2   | 1.959 | 6.54  |
| shift 3      | 1.943 | 6.48  |
| shift 10     | 1.907 | 6.36  |

All of them are within the spread of two seeds, so the defaults are kept; 10 steps is as good with a third of the denoising passes.
