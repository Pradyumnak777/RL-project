#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="${SCRIPT_DIR}/saved_runs"
MAIN_FILE="${SCRIPT_DIR}/main.py"
DATA_UTILS_FILE="${SCRIPT_DIR}/data_utils.py"

usage() {
	echo "Usage: $0 <run_folder_name>"
	echo "Example: $0 exp_lr1e-4_gamma095"
}

if [[ $# -lt 1 ]]; then
	usage
	exit 1
fi

RUN_NAME="$1"
DEST_DIR="${RUNS_ROOT}/${RUN_NAME}"
SUMMARY_FILE="${DEST_DIR}/run_summary.txt"

mkdir -p "${DEST_DIR}"

copied_files=()

copy_if_exists() {
	local src="$1"
	if [[ -f "$src" ]]; then
		cp "$src" "$DEST_DIR/"
		copied_files+=("$(basename "$src")")
	fi
}

extract_assignment() {
	local file_path="$1"
	local key="$2"
	local value
	value="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file_path" | head -n 1 | sed -E "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//")"
	if [[ -z "$value" ]]; then
		echo "NOT_FOUND"
	else
		echo "$value"
	fi
}

copy_if_exists "${SCRIPT_DIR}/training_stats.json"
copy_if_exists "${SCRIPT_DIR}/point_tracker_dqn.pth"

for png_file in "${SCRIPT_DIR}"/plot_*.png; do
	if [[ -f "$png_file" ]]; then
		copy_if_exists "$png_file"
	fi
done

# Fallback: copy all PNGs if no plot_*.png matched
if [[ ${#copied_files[@]} -eq 0 ]] || [[ ! " ${copied_files[*]} " =~ "plot_" ]]; then
	for png_file in "${SCRIPT_DIR}"/*.png; do
		if [[ -f "$png_file" ]]; then
			copy_if_exists "$png_file"
		fi
	done
fi

{
	echo "Run Summary"
	echo "==========="
	echo "Created at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
	echo "Run folder: ${DEST_DIR}"
	echo
	echo "Copied Artifacts"
	echo "----------------"
	if [[ ${#copied_files[@]} -eq 0 ]]; then
		echo "(No matching files found to copy.)"
	else
		for f in "${copied_files[@]}"; do
			echo "- ${f}"
		done
	fi
	echo
	echo "Training Configuration (from main.py)"
	echo "-------------------------------------"
	echo "BATCH_SIZE: $(extract_assignment "$MAIN_FILE" "BATCH_SIZE")"
	echo "GAMMA: $(extract_assignment "$MAIN_FILE" "GAMMA")"
	echo "LR: $(extract_assignment "$MAIN_FILE" "LR")"
	echo "TARGET_UPDATE: $(extract_assignment "$MAIN_FILE" "TARGET_UPDATE")"
	echo "epsilon: $(extract_assignment "$MAIN_FILE" "epsilon")"
	echo "epsilon_min: $(extract_assignment "$MAIN_FILE" "epsilon_min")"
	echo "epsilon_decay: $(extract_assignment "$MAIN_FILE" "epsilon_decay")"
	echo "ReplayBuffer capacity: $(grep -E "ReplayBuffer\(capacity=" "$MAIN_FILE" | head -n1 | sed -E 's/.*capacity=([0-9]+).*/\1/' || true)"
	echo
	echo "Reward Design (from data_utils.py:get_reward)"
	echo "----------------------------------------------"
	awk '
		/^[[:space:]]*def get_reward\(/ {in_block=1}
		in_block {print}
		in_block && /raise ValueError\(f"Invalid action/ {exit}
	' "$DATA_UTILS_FILE"
} > "$SUMMARY_FILE"

echo "Saved run artifacts to: ${DEST_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
