# Run the Workflow by Stage

English · [简体中文](README_zh.md)

Complete command examples are in the root README and `docs/`. There is no script that automatically runs the entire robot workflow in one step. Moving between stages requires placing objects, checking the robot and cameras, uploading data to the cloud GPU server, and letting the on-site operator decide when to enable the model.

Use `./run.sh help` for the unified entry point. Inspect stage-specific arguments with `./run.sh convert --help`, `./run.sh train --help`, `./run.sh serve --help`, and `./run.sh deploy --help`. `configs/*.env.example` contains the actual configuration templates for the two machine roles. Copy them to filenames without `.example`, then fill them in. Do not commit real filesystem paths or SSH configuration to Git.
