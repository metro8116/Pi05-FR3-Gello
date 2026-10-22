# Run the Workflow by Stage

English · [简体中文](README_zh.md)

Complete command examples are in the root README and `docs/`. There is no script that automatically runs the entire robot workflow in one step. Moving between stages requires placing objects, checking the robot and cameras, uploading data to the cloud GPU server, and letting the on-site operator decide when to enable the model.

Use `./run.sh help` for the unified entry point. Inspect stage-specific arguments with `./run.sh convert --help`, `./run.sh train --help`, `./run.sh serve --help`, and `./run.sh deploy --help`. `configs/*.env.example` contains the actual configuration templates for the two machine roles. Copy them to filenames without `.example`, then fill them in. Do not commit real filesystem paths or SSH configuration to Git.

## Bundled examples

| Example | Location | Contents and purpose |
| --- | --- | --- |
| Raw demonstration | [data/raw/avocado_batch01/record_001/](../data/raw/avocado_batch01/record_001/) | One complete schema-2 episode: 304 aligned state/action frames at 15 FPS, both camera streams, saved RGB images, JSON/HDF5, timestamps, and metadata. Use it to inspect the format and check conversion. |
| Deployment recording | [deploy/recordings/20261022_155120_effebe48-177c-4bf9-b394-cbbf5fb0d370_e38d0c5b/](../deploy/recordings/20261022_155120_effebe48-177c-4bf9-b394-cbbf5fb0d370_e38d0c5b/) | Two 640×480 RGB MP4 files, each containing 554 frames at a nominal 30 FPS, plus frame metadata, recording metadata, and encoder logs. Use it to review robot execution. |

These two directories are explicitly tracked by Git. Full datasets, model weights, environments, and new recordings remain excluded by the default ignore rules. Use a new batch name for your own collection, such as `data/raw/avocado_batch02`, to keep the bundled raw example separate.

The deployment video does not contain the collection schema-2 state/action records required for training. It must not be added directly to the demonstration dataset. The single raw example verifies the pipeline; use the [complete LeRobot dataset](https://huggingface.co/datasets/Metroooooooo/fr3-gello-avocado-v30) or your own reviewed demonstrations for task training.

## Validate and convert the raw example

Configure the converter environment as described in the [environment guide](../docs/environments.md), then run these commands from the repository root:

```bash
./run.sh validate \
  --input-dir ./data/raw/avocado_batch01 \
  --episodes record_001 \
  --output-dir ./data/lerobot/fr3_avocado_example_v30 \
  --report ./runs/example_raw_validation.json

./run.sh convert \
  --input-dir ./data/raw/avocado_batch01 \
  --episodes record_001 \
  --repo-id local/fr3_avocado_example \
  --output-dir ./data/lerobot/fr3_avocado_example_v30
```

Validation should select **1 episode / 304 frames**, with zero skipped or invalid episodes. It does not create the converted dataset. Conversion creates a new `fr3_avocado_example_v30` directory and refuses an existing output path; keep it separate from the full training dataset. `--input-dir` points to the batch containing `record_001`, rather than the outer `data/raw` directory.

On a machine with the ML environment configured, check training-side reading without loading model weights or starting training:

```bash
./run.sh check-train \
  --dataset ./data/lerobot/fr3_avocado_example_v30 \
  --data-only
```

See the [data guide](../docs/data.md) for the full data contract, reviewed trimming, and conversion provenance.
