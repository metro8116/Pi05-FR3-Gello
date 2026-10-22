#!/usr/bin/env python3
"""Read-only inventory: never opens serial/camera/robot sockets or imports PI05."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = {"repository": str(ROOT), "hardware_started": False,
              "executables": {name: shutil.which(name) for name in ("uv", "docker", "ssh", "ffmpeg")},
              "profiles": {}}
    for profile, variable in (("robot", "PI05_ROBOT_PYTHON"), ("converter", "PI05_CONVERTER_PYTHON"), ("ml", "PI05_ML_PYTHON")):
        path = Path(os.environ.get(variable, str(ROOT / f".venv-{profile}/bin/python")))
        entry = {"python": str(path), "exists": path.is_file()}
        if entry["exists"]:
            code = "import json,sys,importlib.metadata as m; names=['numpy','opencv-python','h5py','lerobot','torch','transformers','pyrealsense2']; versions={}\nfor name in names:\n try: versions[name]=m.version(name)\n except m.PackageNotFoundError: versions[name]=None\nprint(json.dumps({'version':sys.version.split()[0],'packages':versions}))"
            result = subprocess.run([str(path), "-B", "-c", code], capture_output=True, text=True, timeout=20)
            entry["runtime"] = json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr[-1000:]}
        report["profiles"][profile] = entry
    report["machine_config"] = {name: (ROOT / "configs" / name).is_file() for name in ("local.env", "autodl.env")}
    report["notes"] = ["Missing environments are expected on the other machine.", "This inventory does not prove camera, FCI, GPU or network readiness."]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
