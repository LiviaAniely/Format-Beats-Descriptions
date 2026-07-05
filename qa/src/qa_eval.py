import os
from collections import defaultdict
from pprint import pprint
import pandas as pd

from data_loader import get_dataset
from plot import plot_cross_models, plot_cross_datasets


dataset_map = {
    "logical_fallacy_detection": "logicalfallacy",
    "three_objects": "threeobjects",
    "known_unknowns": "knownunknowns",
}

def get_run_suffix(is_inference=False):
    suffix = os.environ.get("RUN_SUFFIX")
    if suffix is not None:
        return suffix
    
    suffix_file = "../output/current_run_suffix.txt"
    if os.path.exists(suffix_file):
        try:
            with open(suffix_file, "r") as f:
                return f.read().strip()
        except Exception:
            pass
            
    if is_inference:
        os.makedirs("../output", exist_ok=True)
        i = 1
        while True:
            suffix = f"-{i}"
            found = False
            if os.path.exists("../output"):
                for root, dirs, files in os.walk("../output"):
                    for d in dirs:
                        if d.endswith(suffix):
                            found = True
                            break
                    if found:
                        break
            if not found:
                break
            i += 1
        try:
            with open(suffix_file, "w") as f:
                f.write(suffix)
        except Exception:
            pass
        return suffix
        
    i = 1
    latest_suffix = ""
    while True:
        suffix = f"-{i}"
        found = False
        if os.path.exists("../output"):
            for root, dirs, files in os.walk("../output"):
                for d in dirs:
                    if d.endswith(suffix):
                        found = True
                        break
                if found:
                    break
        if found:
            latest_suffix = suffix
            i += 1
        else:
            break
    return latest_suffix

def evaluate_cross_models(dataset_name, output_dir, models=["alpaca-7b", "Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2"]):
    if isinstance(models, str):
        models = [models]

    if dataset_name in dataset_map:
        dataset_name = dataset_map[dataset_name]

    dataset = get_dataset(dataset_name)
    output_dir = dataset.data_dir.replace("data", "output") if output_dir is None else output_dir

    results = defaultdict(dict)

    suffix = get_run_suffix(is_inference=False)
    for model_name in models:
        model_output_dir = f"{output_dir}/{model_name}{suffix}"
        if not os.path.exists(model_output_dir):
            continue
        try:
            output_files = os.listdir(model_output_dir)
        except OSError:
            continue
        output_files = [f"{model_output_dir}/{file}" for file in output_files if file.startswith("4") and ("ensemble_random" in file or "vanilla" in file)]

        for file in output_files:
            # print(f'evaluating {file}')
            if "cot" in file:
                template = file.split(".")[-3] + " (w/ CoT)"
                template = template.replace("ensemble_random", "ERR")
                results[model_name][template] = dataset.evaluate_from_file(file)
            else:
                template = file.split(".")[-2] + " (w/o CoT)"
                template = template.replace("ensemble_random", "ERR")
                results[model_name][template] = dataset.evaluate_from_file(file)

    return results, output_dir.replace("output", "images")

def evaluate_cross_datasets(model, category):
    category_dir = f"../output/{category}"
    if not os.path.exists(category_dir):
        return {}, category_dir.replace("output", "images")
    datasets = os.listdir(category_dir)
    output_dirs = [f"{category_dir}/{dataset}/" for dataset in datasets]

    results = defaultdict(dict)
    for dataset, output_dir in zip(datasets, output_dirs):
        tmp, _ = evaluate_cross_models(dataset, output_dir, model)
        if model in tmp:
            results[dataset] = tmp[model]

    return results, category_dir.replace("output", "images")

def to_excel(results, save_path, sort_order):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        for category, result in results.items():
            df = pd.DataFrame(result).T
            df = df.reindex(sort_order, axis=1)
            df.to_excel(writer, sheet_name=category, float_format="%.16f")
            

if __name__ == "__main__":
    datasets = ["date", "knownunknowns", "logicalfallacy", "threeobjects", "csqa", "strategyqa", "sports", "aqua", "gsm8k"]
    models = ["alpaca-7b", "Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2"]
    sort_order = ["vanilla (w/o CoT)", "vanilla (w/ CoT)", "ERR (w/o CoT)", "ERR (w/ CoT)"]

    task2datasets = {
        "commonsense": ["date", "sports", "csqa", "strategyqa"],
        "logic": ["logicalfallacy", "threeobjects"],
        "math": ["aqua", "gsm8k"],
        "hallucination": ["knownunknowns"]
    }

    # Small models
    suffix = get_run_suffix(is_inference=False)
    small_model_results = {}
    for dataset in datasets:
        dataset_obj = get_dataset(dataset)
        output_dir = dataset_obj.data_dir.replace("data", "output")
        if not os.path.exists(output_dir):
            continue
        results, save_dir = evaluate_cross_models(dataset, None, models=models)
        if results:
            small_model_results[dataset] = results
            plot_cross_models(results, save_path=f"{save_dir}/{dataset}{suffix}.png", title=dataset, sort_order=sort_order)
    if small_model_results:
        to_excel(small_model_results, f"small_models{suffix}.xlsx", sort_order)

    # GPT-3.5
    model = "gpt-3.5-turbo-0125"

    if os.path.exists(f"../output/{model}{suffix}") or any(os.path.exists(f"../output/{cat}") for cat in task2datasets.keys()):
        gpt_results = {}
        for category, datasets in task2datasets.items():
            results, save_dir = evaluate_cross_datasets(model, category)
            if results:
                gpt_results[category] = results
                plot_cross_datasets(results, model, category, save_path=f"{save_dir}/{model}{suffix}.png", sort_order=sort_order)
        if gpt_results:
            to_excel(gpt_results, f"gpt{suffix}.xlsx", sort_order)
