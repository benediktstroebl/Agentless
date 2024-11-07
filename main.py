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
    "--top_n", "3",
    "--compress",
    "--skip_existing",
    "--context_window", "10",
    "--temperature", "0.8",
    "--num_samples", "4"]

# Command 2: Merge
MERGE_ARGS = RAW_ARGS + [
    "--merge",
    "--num_samples", "4",
    "--skip_existing"
]

# Command 3: Repair (Run 1)
REPAIR_RUN1_ARGS = RAW_ARGS + [
    "--loc_interval",
    "--top_n", "3",
    "--context_window", "10",
    "--max_samples", "21",
    "--cot",
    "--diff_format",
    "--gen_and_process"]

# Command 4: Repair (Run 2)
REPAIR_RUN2_ARGS = RAW_ARGS + [
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
    "--num_samples", "42",
    "--deduplicate",
    "--plausible"
]


def run_agentless(tasks: dict[str, dict], **kwargs) -> dict[str, str]:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = kwargs['model']

    output_folder = f"results_run_agentless"

    output_folder = os.getcwd() + "/" + output_folder

    assert len(tasks) == 1, "Expected a single task for this benchmark, got {len(tasks)}"

    input = {}
    input['problem_statement'] = tasks[list(tasks.keys())[0]]['input']
    input['instance_id'] = tasks[list(tasks.keys())[0]]['id']
    input['repo'] = tasks[list(tasks.keys())[0]]['metadata']['repo']
    input['base_commit'] = tasks[list(tasks.keys())[0]]['metadata']['base_commit']
    input = [input]

    assert os.getcwd() == os.path.abspath(os.path.dirname(__file__)), "Please run this script from the root directory of the project"
    # generate edit locations
    sys.argv = LOCALIZE_ARGS + ["--model",  model, "--output_folder", f"{output_folder}/location"]
    if "claude" in model:
        sys.argv += ["--backend", "anthropic"]
    localize(benchmark_data=input)

    assert os.getcwd() == os.path.abspath(os.path.dirname(__file__)), "Please run this script from the root directory of the project"

    # Merge 
    sys.argv = MERGE_ARGS + ["--output_folder", f"{output_folder}/location_merged", "--start_file", f"{output_folder}/location/loc_outputs.jsonl"]
    if "claude" in model:
        sys.argv += ["--backend", "anthropic"]
    localize()

    assert os.getcwd() == os.path.abspath(os.path.dirname(__file__)), "Please run this script from the root directory of the project"

    # Repair Run 1
    sys.argv = REPAIR_RUN1_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_0-1_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_1"]
    if "claude" in model:
        sys.argv += ["--backend", "anthropic"]
    repair(benchmark_data=input)

    assert os.getcwd() == os.path.abspath(os.path.dirname(__file__)), "Please run this script from the root directory of the project"

    # Repair Run 2
    sys.argv = REPAIR_RUN2_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_2-3_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_2"]
    if "claude" in model:
        sys.argv += ["--backend", "anthropic"]
    repair(benchmark_data=input)

    assert os.getcwd() == os.path.abspath(os.path.dirname(__file__)), "Please run this script from the root directory of the project"

    # Rerank
    sys.argv = RERANK_ARGS + ["--patch_folder", f"{output_folder}/repair_run_1,{output_folder}/repair_run_2"]
    rerank()

    # assert that the current working directory is the directory where this file is located
    assert os.getcwd() == os.path.abspath(os.path.dirname(__file__)), "Please run this script from the root directory of the project"
    

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

    result = {}
    for i, instance in enumerate(input):
        result[instance['instance_id']] = instance['model_patch']

    return result

def run_agentless_gpt_4o_mini_c_1(input):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"results_run_agentless_gpt_4o_mini_c_1"
    model = "gpt-4o-mini-2024-07-18"

    # generate edit locations
    sys.argv = LOCALIZE_ARGS + ["--model",  model, "--output_folder", f"{output_folder}/location"]
    localize(benchmark_data=input)

    # Merge 
    sys.argv = MERGE_ARGS + ["--output_folder", f"{output_folder}/location_merged", "--start_file", f"{output_folder}/location/loc_outputs.jsonl"]
    localize()

    # Repair Run 1
    sys.argv = REPAIR_RUN1_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_0-1_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_1"]
    repair(benchmark_data=input)

    # Repair Run 2
    sys.argv = REPAIR_RUN2_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_2-3_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_2"]
    repair(benchmark_data=input)

    # Rerank
    sys.argv = RERANK_ARGS + ["--patch_folder", f"{output_folder}/repair_run_1,results/repair_run_2"]
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

    

    return input


def run_agentless_Meta_Llama_3_1_8B_c_1(input):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    output_folder = f"results_run_agentless_Meta_Llama_3_1_8B_c_1"

    # generate edit locations
    sys.argv = LOCALIZE_ARGS + ["--model",  model, "--output_folder", f"{output_folder}/location"]
    localize(benchmark_data=input)

    # Merge 
    sys.argv = MERGE_ARGS + ["--output_folder", f"{output_folder}/location_merged", "--start_file", f"{output_folder}/location/loc_outputs.jsonl"]
    localize()

    # Repair Run 1
    sys.argv = REPAIR_RUN1_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_0-1_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_1"]
    repair(benchmark_data=input)

    # Repair Run 2
    sys.argv = REPAIR_RUN2_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_2-3_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_2"]
    repair(benchmark_data=input)

    # Rerank
    sys.argv = RERANK_ARGS + ["--patch_folder", f"{output_folder}/repair_run_1,results/repair_run_2"]
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

    os.rename(output_folder, f"{output_folder}_{now}")

    return input


def run_agentless_Meta_Llama_3_1_70B_c_1(input):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    
    output_folder = f"results_run_agentless_Meta_Llama_3_1_70B_c_1"

    # generate edit locations
    sys.argv = LOCALIZE_ARGS + ["--model",  model, "--output_folder", f"{output_folder}/location"]
    localize(benchmark_data=input)

    # Merge 
    sys.argv = MERGE_ARGS + ["--output_folder", f"{output_folder}/location_merged", "--start_file", f"{output_folder}/location/loc_outputs.jsonl"]
    localize()

    # Repair Run 1
    sys.argv = REPAIR_RUN1_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_0-1_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_1"]
    repair(benchmark_data=input)

    # Repair Run 2
    sys.argv = REPAIR_RUN2_ARGS + ["--model",  model, "--loc_file", f"{output_folder}/location_merged/loc_merged_2-3_outputs.jsonl", "--output_folder", f"{output_folder}/repair_run_2"]
    repair(benchmark_data=input)

    # Rerank
    sys.argv = RERANK_ARGS + ["--patch_folder", f"{output_folder}/repair_run_1,results/repair_run_2"]
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

    os.rename(output_folder, f"{output_folder}_{now}")

    return input



