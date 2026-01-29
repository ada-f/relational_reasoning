#!/usr/bin/env bash
set -euo pipefail
cat data_parts/total_results_all_models_part_* > total_results_all_models.csv
echo "Rebuilt total_results_all_models.csv"
