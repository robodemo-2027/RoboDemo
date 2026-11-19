"""Run the release CPU suites inside the built image, without model assets or GPUs."""
import json
from pathlib import Path
import re
import subprocess


def validate(output_dir=Path(".")):
    report_path = output_dir / "container-regression.json"
    report = {"image": "robodemo-evaluation:cu124", "gpu_execution": False, "variants": {}}
    for variant in ("human", "robot"):
        command = [
            "docker", "run", "--rm", "--entrypoint", "python",
            "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", f"ROBODEMO_TEST_VARIANT={variant}",
            "-v", f"{Path.cwd()}:/workspace/RoboDemo:ro",
            report["image"], "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"CPU variant: {variant}\n{result.stdout}", flush=True)
        counts = re.findall(r"(\d+) passed", result.stdout)
        row = {"exit_code": result.returncode, "passed_tests": int(counts[-1]) if counts else 0}
        report["variants"][variant] = row
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode or not row["passed_tests"]:
            raise RuntimeError(f"The image's {variant} CPU regression suite did not pass.")
    report["passed"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    validate()
