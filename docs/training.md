# FR3 avocado task: data, π0.5 fine-tuning, and checkpoint recovery

English · [简体中文](training_zh.md)

This repository calls the **official LeRobot 0.6.1 trainer**. The official implementation continues to handle data sampling, Accelerate, the optimizer, the learning-rate schedule, and checkpoint saving and recovery. This project adapts only the FR3 data contract, official processors, and strict weight loading.

The repository contains code, configuration, and tests. It **does not include raw demonstrations, a cleaned dataset, `pi05_base` weights, or training checkpoints**. Prepare these assets separately or reference their existing paths. The training commands in this guide run only when the operator explicitly executes them; organizing the repository, checking the environment, and using `--dry-run` do not start training.

Run the following commands from the repository root. The new layout uses `data/lerobot/fr3_avocado_v30` for the dataset and `outputs/` for training output. See the [dependency and conversion guide](../requirements/README.md) for collection and conversion, and the [deployment guide](deployment.md) for using a trained model.

## 1. Select the ML environment on the cloud GPU server

To reuse an existing uv environment that has already trained successfully, no installation or `setup ml` command is needed:

```bash
cd /root/autodl-tmp/Pi05-FR3-Gello
export PI05_ML_PYTHON="/root/autodl-tmp/mujoco_pick_place_pi05/.venv/bin/python"
"$PI05_ML_PYTHON" -c 'import sys, importlib.metadata as m; print(sys.executable); print("LeRobot:", m.version("lerobot"))'
```

You can also set `PI05_ML_PYTHON` in the untracked `configs/autodl.env` file. `run.sh` reads the private environment files; if the variable is already set in one of those files, that file's value takes precedence. The entry point runs this repository's source with the selected Python interpreter. Installing this repository into the old environment is not required.

The working ML combination is Python 3.12, LeRobot 0.6.1, Torch 2.11.0+cu128, and Transformers 5.5.4. For a new isolated environment, follow the environment guide linked from the root README to create `.venv-ml`. Do not install robot or converter dependencies into an old environment that is currently training. The converter also uses LeRobot 0.6.1, but installs only CPU data-processing dependencies; training uses CUDA and PI05 model dependencies. Keep these environments separate.

Preserve the existing `HF_HOME`, `HF_HUB_CACHE`, and tokenizer assets. If the cloud GPU server already uses the following cache locations, you can continue using them:

```bash
export HF_HOME="/root/autodl-tmp/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
```

## 2. Prepare the complete cleaned dataset separately

The default training dataset path is:

```text
Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30/
├── meta/       # Info, tasks, statistics, episode metadata, and conversion provenance
├── data/       # Scalar data in Parquet files
└── videos/     # Both camera streams
```

Keep the entire exported directory, including `meta/conversion_manifest.json` or a compatible provenance manifest, statistics, source timestamps, and trimming records. Copying only MP4 files, Parquet files, or `info.json` is insufficient. Training does not require the original recording directory to remain at its old location; historical absolute source paths in the manifest are used only for traceability.

An existing cleaned v3.0 dataset does not need another cleaning or conversion pass. You can copy it separately into the new repository's data directory on the local computer. The example below refuses an existing destination:

```bash
CLEAN_DATASET="/home/zezhi/桌面/Pi05-train/data/fr3_avocado_v30"
NEW_DATASET="$PWD/data/lerobot/fr3_avocado_v30"
if [[ ! -f "$CLEAN_DATASET/meta/info.json" ]]; then
  echo "The complete source dataset was not found; check the path first." >&2
elif [[ -e "$NEW_DATASET" || -L "$NEW_DATASET" ]]; then
  echo "The destination exists; keep it and choose a new dataset name." >&2
else
  mkdir -p "$(dirname "$NEW_DATASET")" && cp -a -- "$CLEAN_DATASET" "$NEW_DATASET"
fi
```

To upload the dataset separately from the local computer to the cloud GPU server, replace the SSH placeholders with your own connection settings. The destination must be a new, nonexistent path; do not merge a new batch into an existing dataset:

```bash
# Run on the local computer; CLEAN_DATASET is the complete v3.0 directory above.
ssh -p YOUR_SSH_PORT root@YOUR_CLOUD_GPU_HOST \
  "mkdir -p /root/autodl-tmp/Pi05-FR3-Gello/data/lerobot && \
   test ! -e /root/autodl-tmp/Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30 && \
   test ! -L /root/autodl-tmp/Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30" && \
rsync -a --progress -e 'ssh -p YOUR_SSH_PORT' \
  "$CLEAN_DATASET/" \
  root@YOUR_CLOUD_GPU_HOST:/root/autodl-tmp/Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30/
```

Confirm that the first SSH check succeeds before uploading. After an interrupted transfer, resume only the same batch into this newly created destination. Neither original data nor models should be committed to Git. If the dataset is already on the cloud GPU server, for example in the previous project's `data/fr3_avocado_v30`, pass that absolute path to `--dataset` directly instead of copying it.

The task uses the following fixed interface:

| Input or output | Meaning |
| --- | --- |
| `observation.state`, 8 dimensions | Seven measured FR3 joint angles in rad; the eighth value is the measured total gripper opening in m. |
| `action`, 8 dimensions | Seven absolute FR3 joint-position targets; the eighth value is the most recently accepted total gripper-opening target in m. |
| `observation.images.camera_1/2` | Two 640×480 RGB streams in a fixed order; the official reader supplies floating-point CHW images. |
| Frequency and task | 15 FPS; `Put the avocado in the basket.` |
| Gripper control | Move width targets, matching collection. The model does not predict force, and the measured width after contact with an object is not substituted for the closing target. |

The model processors handle image resizing, task tokenization, and normalization of state/action values using training statistics. State and action use quantile statistics over all training rows. The saved processors accompany each checkpoint for reuse during inference. Action chunks contain 50 steps. The official reader pads the end of each episode rather than concatenating future actions from another episode.

## 3. Point to an existing complete `pi05_base` snapshot

Start a new training run from `lerobot/pi05_base`, rather than weights fine-tuned on the previous simulation task. The path must identify a specific snapshot containing both `config.json` and one complete `model.safetensors` file:

```bash
export PI05_BASE_MODEL="/root/autodl-tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/YOUR_EXISTING_SNAPSHOT_HASH"
test -f "$PI05_BASE_MODEL/config.json"
test -f "$PI05_BASE_MODEL/model.safetensors"
```

Replace `YOUR_EXISTING_SNAPSHOT_HASH` with the actual downloaded directory name. Do not point only to the outer `models--lerobot--pi05_base` directory: it normally contains `refs/`, `snapshots/`, and `blobs/`, and is not itself a loadable model directory. If `refs/main` exists, inspect the hash it references and select the corresponding snapshot. The code reads offline and does not download missing weights for you.

By default, the tokenizer is read from the existing cache for `google/paligemma-3b-pt-224`. When using a separate tokenizer directory, add `--tokenizer /absolute/path/to/tokenizer` to **both the checking and new-training commands**. Checkpoint processors save the tokenizer source; they do not automatically bundle the entire tokenizer cache.

## 4. Check the data, environment, and configuration

On the cloud GPU server, run:

```bash
./run.sh check-train \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --base-model "$PI05_BASE_MODEL"
```

The checks cover the data contract, all scalar rows and statistics, representative-frame decoding for both cameras, 50-step action chunks and final-frame padding, the ML API, CUDA/bfloat16 support, weight-file headers, and the local tokenizer. A successful result includes `status: passed`, `validation: false`, and `episode_selection: all`. This check does not construct the complete π0.5 network; strict matching of all parameters to the model structure occurs when the model is actually loaded.

For data-only checks, use `./run.sh check-train --dataset /path/to/dataset --data-only`. This still requires data dependencies such as PyAV and the LeRobot reader, but does not require base-model weights or CUDA.

Inspect the final arguments without loading the model or creating an output directory:

```bash
./run.sh train \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --policy-path "$PI05_BASE_MODEL" \
  --out ./outputs/fr3_avocado_run2 \
  --wandb true --dry-run
```

`--policy-path` and `--base-model`, `--dataset` and `--data-root`, and `--out` and `--output-dir` are pairs of aliases. All relative data/output arguments are resolved against the working directory in which the command runs. Use absolute paths when calling from another directory.

## 5. Run a two-step smoke test and one recovery check first

These commands actually use the GPU, train, and save complete model/optimizer checkpoints. The operator should run them explicitly when a GPU is available, with output directories that do not already exist. Avoid unintentionally overlapping an inference server or another training job on the same GPU; the entry point does not stop other processes automatically.

```bash
./run.sh train \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --policy-path "$PI05_BASE_MODEL" \
  --out ./outputs/smoke_2steps \
  --steps 2 --batch-size 1 --save-freq 1 --log-freq 1 \
  --gradient-checkpointing false --wandb false --offline
```

After confirming that `000001` and `000002` were saved, resume from step 1 into another new output directory:

```bash
./run.sh resume \
  --resume-from ./outputs/smoke_2steps/checkpoints/000001 \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --out ./outputs/smoke_resume
```

This preserves the smoke configuration's **total of 2 steps**, so only step 2 remains. Recovery reads the checkpoint's model, processors, optimizer, scheduler, RNG state, and official training-step counter. It does not initialize again from the base model. Do not supply `--policy-path` for this smoke resume or change the total step count.

## 6. Full training: 20k steps, batch 16, no gradient checkpointing, no validation split

If using W&B, first run `"$PI05_ML_PYTHON" -m wandb login` in the selected ML environment, or reuse an existing login. Do not include or record an API key in the training command.

```bash
./run.sh train \
  --policy-path "$PI05_BASE_MODEL" \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --out ./outputs/fr3_avocado_run2 \
  --offline \
  --gradient-checkpointing false \
  --batch-size 16 --num-workers 8 \
  --steps 20000 --save-freq 2000 --log-freq 10 \
  --wandb true
```

Choose a new experiment directory, such as `fr3_avocado_run2`. A new training run refuses to overwrite an existing output directory. To continue an existing experiment, use `resume` as described in the next section. Pass `--wandb false` if W&B is not needed.

| Parameter | Setting for this task |
| --- | --- |
| Total optimizer steps / batch size | 20,000 / 16 |
| DataLoader workers / seed | 8 / 1000 |
| Save / logging interval | Every 2,000 / 10 steps |
| dtype / gradient checkpointing | bfloat16 / false |
| Freeze vision encoder / train expert only | true / false |
| Chunk size / n action steps | 50 / 50 |
| AdamW | lr 2.5e-5, betas (0.9, 0.95), weight decay 0.01, clip norm 1.0 |
| Official π0.5 scheduler | warmup 1000, decay 30000, final lr 2.5e-6 |
| Data selection / validation | All episodes; no validation split, validation loss, or simulation evaluation. |

The learning-rate schedule uses the original official preset; its decay range is not the same as this run's total update count. The project adds no custom training loop or gradient-accumulation implementation, and `--gradient-accumulation-steps` accepts only 1. W&B records metrics with model-artifact uploading disabled. Checkpoints are not uploaded to Hugging Face either.

## 7. Continue existing training or move experiment paths

To continue the same experiment within this repository:

```bash
./run.sh resume \
  --resume-from ./outputs/fr3_avocado_run2/checkpoints/last
```

`last` means the most recently saved checkpoint, not the model with the highest success rate. You can also select a numbered directory, such as `.../checkpoints/008000`. When resuming toward a total of 20,000 steps, a checkpoint at step 8,000 has 12,000 updates remaining; it **does not run another 20,000 steps**. Do not pass `--steps 12000`.

An old project's environment, data, and checkpoints can all stay in their original locations. For example, use this repository's code to continue the original experiment directly:

```bash
export PI05_ML_PYTHON="/root/autodl-tmp/mujoco_pick_place_pi05/.venv/bin/python"
./run.sh resume \
  --resume-from /root/autodl-tmp/Pi05-train/outputs/fr3_avocado_pi05/checkpoints/last
```

This command uses the absolute data and output paths stored in the checkpoint. Subsequent checkpoints continue to be written to the original experiment's output directory. This is an explicit operator choice to resume training, rather than copying or committing the old experiment into the new repository.

If the data or subsequent output really needs to move, specify the paths explicitly while preserving the data contents and experiment settings, including batch size, total steps, and learning rate:

```bash
./run.sh resume \
  --resume-from /path/to/original-run/checkpoints/008000 \
  --dataset /path/to/relocated/fr3_avocado_v30 \
  --out /path/to/new/resumed-run
```

A move changes only asset/output locations; it should not substitute a new task or a new batch of data. The saved W&B settings and run ID are preserved by default; do not supply an additional new project/run. Recovery permits adjustments to runtime settings such as worker count and logging/save intervals, but this guide leaves them unchanged. Official sampling and recovery still depend on conditions such as the original batch/world size, so bitwise-identical results across hardware are not guaranteed.

Recovery requires the entire numbered checkpoint:

```text
008000/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train_config.json
│   ├── policy_preprocessor.json
│   ├── policy_postprocessor.json
│   └── statistics/state files referenced by the processors
└── training_state/
    ├── optimizer_param_groups.json
    ├── optimizer_state.safetensors
    ├── scheduler_state.json
    ├── rng_state.safetensors
    └── training_step.json
```

Deployment primarily uses the complete `pretrained_model/` directory. **Resuming training also requires its sibling `training_state/` directory.** The old custom loop's `COMPLETE.json` / `training_state/state.pt` are not equivalent to official checkpoints and are rejected by the entry point. The tokenizer's saved cache ID or absolute local path must remain available; resume does not accept `--tokenizer` to replace it.

## 8. Native-checkpoint loading fix and common errors

Base-model initialization and training recovery use two explicit strict-loading paths:

- A new training run can adapt `pi05_base` using the official base-weight key migration.
- `resume` reads an official native checkpoint and strictly loads the original saved names. It does not strip `model.` prefixes or call the base-model remapper.

This preserves `lm_head.weight` and `embed_tokens.weight` when training has made their values different. Missing aliases may be completed only when the current model actually uses **the same Parameter/buffer object**. Missing independent parameters, unexpected keys, incorrect shapes, and conflicts in shared parameters still fail. Loading never falls back to a random model.

The previous `Conflicting remapped checkpoint key: ...embed_tokens.weight` error resulted from applying base-model remapping rules to a training checkpoint again. The current resume branch fixes this. Do not work around the error by deleting checkpoint keys, forcibly tying the two weights, or disabling strict checks. The fix does not change the official optimizer, sampler, saving, or recovery mechanisms, and does not rewrite existing checkpoints.

| Message or condition | Action |
| --- | --- |
| `config.json` is missing from the base-model directory | Use a complete `snapshots/<hash>` directory, not the outer Hub cache directory. |
| Tokenizer / processor state is missing | Preserve all processor files and the tokenizer cache/directory used during training. |
| Data statistics do not cover all training frames | Use a correctly exported dataset without a validation split and pass `check-train` first. Do not edit statistics files merely to make the check pass. |
| Recovery requires preserving steps / batch settings | Remove arguments that alter the experiment. Create a new experiment when new hyperparameters are needed. |
| CUDA OOM | Check other processes on the same GPU first; the program does not terminate them for you. Changing batch size or gradient checkpointing changes the experiment settings and must not be silently mixed into the original resumed run. |
| Continuing after Ctrl+C | Resume from the most recently saved official checkpoint. `last` cannot automatically recover updates that were never saved. |
