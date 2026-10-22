# Data interface, backups, and LeRobot v3.0 conversion

English · [简体中文](data_zh.md)

The default paths are `data/raw/record_NNN` → `data/lerobot/fr3_avocado_v30`. Both directories are ignored by Git; actual demonstration data is not included in the repository. Camera recordings, JSON, HDF5, timestamps, and metadata together form one recording. A subset of those files is not sufficient for conversion.

## Fixed interface

| Item | Contract |
| --- | --- |
| task | `Put the avocado in the basket.` |
| `observation.state` | Measured FR3 q1…q7 in rad + measured total gripper width in m |
| `action` | Absolute q1…q7 targets from the FR3 controller in rad + the most recently accepted total gripper-width target in m |
| state/action | float32, 8 dimensions each; neither raw GELLO readings nor a 0–1 gripper-closure ratio |
| camera_1 | `observation.images.camera_1`, serial `233522074047` |
| camera_2 | `observation.images.camera_2`, serial `405622074065` |
| Images | RGB 640×480; loaded after conversion as floating-point CHW values in [0,1] |
| FPS | RGB capture at 30, recording/training at 15, with no implicit resampling |
| Action mode | absolute, `use_relative_actions=false` |
| Current gripper execution | Move at 0.04 m/s, total width 0–0.08 m; the model predicts neither force nor speed |

The teleoperation controller uses the current FR3/GELLO poses as its relative zero reference, but records the **resulting absolute FR3 commands**. Deployment must not add the initial offset to predictions a second time. A difference between measured and target total gripper widths is expected when grasping an object. Do not replace the target with the measured width to conceal that difference.

## Back up before reviewing

Close the recording and teleoperation programs, confirm that files are no longer being written, and make a separate copy of the original directory in a location with sufficient space. Do not overwrite an old backup with the same destination. Preserve original metadata and logs for traceability. You can use `cp -a` or `rsync -a` after checking the source and destination. A backup under `data/` is also excluded from Git.

Review both MP4 streams episode by episode. Check whether the object stays grasped, is actually released at the destination, and whether the arm completes its retreat. Also inspect the actions, timestamps, and camera identifiers. Mark episodes that failed in the real scene as failure/aborted; complete files alone do not make them successful demonstrations. A single gripper Move `Command failed!` message does not establish whether an entire episode is usable: the converter preserves original execution diagnostics, and task success still requires video and measurement evidence.

Leading and trailing nearly stationary frames may be trimmed **after manual review**, while preserving the full approach, grasp, placement, gripper opening, and retreat. Waiting in the middle may be part of contact or release behavior; small action changes alone are not grounds for automatically discarding frames. This repository does not automatically delete or modify original files.

## Read-only validation

```bash
./run.sh validate --input-dir ./data/raw \
  --report ./runs/raw_validation.json
```

The entry point automatically adds `--dry-run`. Validation checks the schema, JSON/HDF5 consistency, eight-dimensional data, completeness/task/labels, source-data freshness, consistent contents for the same command identity, camera ordering, video frame counts, exposure/receive times, FPS, and frame gaps. Missing dependencies cause a failure; “Selected 0” must not be treated as a successful check.

By default, only recordings with `status=complete`, `outcome=success`, and a matching interface are selected. Invalid segments are not silently repaired and included. The result must contain a nonempty selected set and a positive frame count. Read the report's skipped/invalid reasons before deciding whether to exclude or re-record an episode.

## Export

```bash
./run.sh convert --input-dir ./data/raw \
  --repo-id local/fr3_avocado_pi05 \
  --output-dir ./data/lerobot/fr3_avocado_v30
```

An existing output directory is never overwritten. Export first writes to temporary staging and publishes a new directory only on success; original files remain unchanged. The default is `--val-ratio 0`, so all valid episodes are used for training in this task. Use `--episodes record_001 record_002` for an explicit selection. Changing the selection creates a new dataset version; do not replace a dataset in place after training has started.

The output contains Parquet data, both video streams, task/episode/statistics metadata, a conversion manifest, source-file digests, and per-frame source timestamps. The training interface reads these provenance conventions and independently checks statistics over all scalar rows. Keep the **entire output directory** together; do not move or separately re-encode its videos.

Conversion and training both pin LeRobot 0.6.1, while the disk format is v3.0. Conversion uses CPU PyTorch and training on the cloud GPU server uses CUDA PyTorch, so the environments remain separate. See [requirements/README.md](../requirements/README.md).

<a id="manual-trim-plan"></a>

## Optional manually reviewed trimming plan

The converter supports `--trim-plan /absolute/path/to/plan.json`. A plan uses schema `fr3_tail_trim_v1` and fps=15. It must contain exactly one entry for each selected record, specifying `original_frames`, `keep_start_inclusive`, `keep_end_exclusive`, nonempty `reason` and `review` fields, and a complete `source_files` fingerprint dictionary with size/mtime_ns/sha256 for every file. Intervals include their starting frame and exclude their ending frame.

`source_files` must describe the complete original files as they existed during review, rather than digests of already-trimmed files. The tool validates the entire original recording first, then slices state, actions, images, and source timestamps, recomputes statistics, and records original frame indices and the trim plan. Conversion is refused if the plan does not cover every selected recording, fingerprints do not match, or a retained interval is too short. Even untrimmed entries must explicitly retain [0,original_frames).

No new trim plan was created for existing real recordings during this work, and they were not cleaned again. An existing cleaned v3.0 dataset can be copied or passed directly to training; its directory does not need a clean suffix. See the [training guide](training.md) for separate copying and upload instructions.

## Checks after conversion

Run in the ML environment:

```bash
./run.sh check-train --dataset ./data/lerobot/fr3_avocado_v30 --data-only
```

This checks training-side reading, samples from both cameras, 50-step action chunks, and episode-end padding, while checking that statistics respect episode boundaries. Start training only after validation passes. Raw recordings can stay on the robot computer and do not need uploading to the cloud GPU server. Training requires the complete v3.0 dataset and model/tokenizer assets.

Automatically saved deployment-camera videos are for reviewing execution and are stored separately in the deployment recording directory. They lack the collection schema-2 state/action data and causal alignment and must not be added directly to the demonstration dataset.
