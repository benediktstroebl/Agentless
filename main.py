from agentless.fl.localize import main as localize
from agentless.repair.repair import main as repair
from agentless.repair.rerank import main as rerank
import os
import sys
import json
from datetime import datetime

# Original sys.argv (assuming it contains only the script name)
RAW_ARGS = sys.argv[:1]

# Command 1: Localize
LOCALIZE_ARGS = RAW_ARGS + [
    "--file_level",
    "--related_level",
    "--fine_grain_line_level",
    "--output_folder", "results/location",
    "--top_n", "3",
    "--compress",
    "--context_window", "10",
    "--temperature", "0.8",
    "--num_samples", "4"]

# Command 2: Merge
MERGE_ARGS = RAW_ARGS + [
    "--merge",
    "--output_folder", "results/location_merged",
    "--start_file", "results/location/loc_outputs.jsonl",
    "--num_samples", "4"
]

# Command 3: Repair (Run 1)
REPAIR_RUN1_ARGS = RAW_ARGS + [
    "--loc_file", "results/location_merged/loc_merged_0-1_outputs.jsonl",
    "--output_folder", "results/repair_run_1",
    "--loc_interval",
    "--top_n", "3",
    "--context_window", "10",
    "--max_samples", "21",
    "--cot",
    "--diff_format",
    "--gen_and_process"]

# Command 4: Repair (Run 2)
REPAIR_RUN2_ARGS = RAW_ARGS + [
    "--loc_file", "results/location_merged/loc_merged_2-3_outputs.jsonl",
    "--output_folder", "results/repair_run_2",
    "--loc_interval",
    "--top_n", "3",
    "--context_window", "10",
    "--max_samples", "21",
    "--cot",
    "--diff_format",
    "--gen_and_process"
]

# Command 5: Rerank
RERANK_ARGS = RAW_ARGS + [
    "--patch_folder", "results/repair_run_1,results/repair_run_2",
    "--num_samples", "42",
    "--deduplicate",
    "--plausible"
]

def run_agentless_gpt_4o_mini_c_1(input):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # output_folder = f"results_run_agentless_gpt_4o_mini_c_1_{now}"
    output_folder = f"results_run_agentless_gpt_4o_mini_c_1_2024-08-17_13-53-01"

    # generate edit locations
    sys.argv = LOCALIZE_ARGS + ["--model",  "gpt-4o-mini-2024-07-18"]
    localize(benchmark_data=input)

    # Merge
    sys.argv = MERGE_ARGS
    localize()

    # Repair Run 1
    sys.argv = REPAIR_RUN1_ARGS + ["--model",  "gpt-4o-mini-2024-07-18"]
    repair(benchmark_data=input)

    # Repair Run 2
    sys.argv = REPAIR_RUN2_ARGS + ["--model",  "gpt-4o-mini-2024-07-18"]
    repair(benchmark_data=input)

    # Rerank
    sys.argv = RERANK_ARGS
    rerank()

    # load all_preds.jsonl file in trajectories and add key data to input
    with open('all_preds.jsonl', "r") as f:
        data = list(f)

    # for each instance_id in input dict, find the corresponding prediction in data
    # and add it to the input dict
    for i, instance in enumerate(input):
        for line in data:
            line = json.loads(line)
            if instance['instance_id'] == line['instance_id']:
                instance['model_name_or_path'] = line['model_name_or_path']
                instance['model_patch'] = line['model_patch']
                break
            instance['model_name_or_path'] = "agentless"
            instance['model_patch'] = "No patch returned"

    # rename results folder
    os.rename("results_run_agentless_gpt_4o_mini_c_1_2024-08-17_13-53-01", output_folder)

    return input



