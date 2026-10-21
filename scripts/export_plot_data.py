"""Export plotting datasets for R and JavaScript from verified raw evidence."""

import argparse
from pathlib import Path

from hpc_ai_perf.analysis.plot_data import export_plot_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result-root", type=Path, default=Path("results/raw"))
parser.add_argument("--output-dir", type=Path, default=Path("results/plot_data"))
parser.add_argument("--execution-matrix", type=Path, default=Path("platforms/execution_matrix.json"))
args = parser.parse_args()
manifest = export_plot_data(args.result_root, args.output_dir, args.execution_matrix)
print(f"Exported {len(manifest['datasets'])} CSV/JSON dataset pairs to {args.output_dir}")
print(
    f"{manifest['source_records']} source records; {manifest['current_performance_trials']} current performance trials"
)
