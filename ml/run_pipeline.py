"""Run the whole ML pipeline, in order, from the project root:

    python ml/run_pipeline.py            # everything (about 2-3 minutes)
    python ml/run_pipeline.py --skip-data   # keep the existing CSVs, redo selection onward

Stages
  1. generate_data    synthetic world (deterministic, seed 42)       -> ml/data/*.csv
  2. data_loader      validate the data, write the data report       -> ml/data/data_report.json
  3. select_models    rolling-origin model selection                 -> ml/models/selection.json
  4. train            fit + calibrate + one test evaluation          -> ml/models/*.pkl, metrics.json
  5. diagnostics      over/underfitting checks, fresh worlds, controls -> ml/models/diagnostics.json

Then start the dashboard:  uvicorn backend.main:app --reload
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ML = Path(__file__).resolve().parent
STAGES = [("generate_data.py", "1/5 generate data"), ("data_loader.py", "2/5 validate data"),
          ("select_models.py", "3/5 select models"), ("train.py", "4/5 train and evaluate"),
          ("diagnostics.py", "5/5 over/underfitting diagnostics")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true", help="reuse the existing CSVs")
    args = ap.parse_args()
    t0 = time.time()
    for script, label in STAGES:
        if args.skip_data and script == "generate_data.py":
            continue
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
        r = subprocess.run([sys.executable, str(ML / script)], cwd=ML.parent)
        if r.returncode != 0:
            sys.exit(f"stage failed: {script}")
    print(f"\nPipeline finished in {time.time() - t0:.0f}s. Start the dashboard with:\n"
          f"    uvicorn backend.main:app --reload")


if __name__ == "__main__":
    main()
