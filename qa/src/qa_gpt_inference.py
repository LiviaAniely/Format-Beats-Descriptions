import os
from tqdm import tqdm
from typing import Union

from openai import OpenAI

from data_loader import get_dataset, ReasoningData
from template import get_prompt_generator


# Replace with the path to the file containing random nouns
PATH_TO_RAND_NOUNS = "./noun/noun_65536.txt"

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

def encapsulate_prompt(prompt):
    messages = [
        {"role": "user", "content": f"{prompt}"},
    ]
    return messages

def main(
        dataset_name, 
        model_name="gpt-3.5-turbo-0125", 
        shot=4, 
        templates=["vanilla", "ensemble_random"], 
        is_cot: Union[bool, str]=False,
        max_new_tokens=256,
        num_nouns=2,
        **kwargs
):
    
    if is_cot == "both":
        main(dataset_name, model_name, shot, templates, True, max_new_tokens, num_nouns, **kwargs)
        is_cot = False
    
    if isinstance(templates, str):
        templates = [templates]
        
    client = OpenAI(
        base_url = "https://api.openai.com/v1",
        api_key="sk-",
    )
                
    dataset: ReasoningData = get_dataset(dataset_name, **kwargs)
    output_dir = dataset.data_dir.replace("data", "output")

    examples, (test_inputs, gold_answers) = dataset.load_data(is_cot=is_cot, shot=shot)

    suffix = get_run_suffix(is_inference=True)
    output_dir = f"{output_dir}/{model_name}{suffix}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for template in templates:
        output_fn = f"{output_dir}/{shot}.{template}.txt" if not is_cot \
        else f"{output_dir}/{shot}.{template}.cot.txt"

        if "single" in template:
            num_nouns = 1

        if "ensemble" in template or "single" in template:
            with open(PATH_TO_RAND_NOUNS, "r") as f:
                rand_nouns = f.readlines()
            rand_nouns = [noun.strip() for noun in rand_nouns]

        with open(output_fn, "w") as f:
            prompt_generator = get_prompt_generator(template)
            prompts = []
            for idx, test_input in tqdm(enumerate(test_inputs), ncols=60):
                if "ensemble" in template or "single" in template:
                    curr_nouns = rand_nouns[idx * num_nouns: (idx + 1) * num_nouns]
                    ensemble_args = {"before_test_input": dataset.before_test_input, "rand_nouns": curr_nouns}
                    prompt = prompt_generator(dataset.instruction, test_input, examples, shot=shot, **ensemble_args)
                else:
                    prompt = prompt_generator(dataset.instruction, test_input, examples, shot=shot, before_test_input=dataset.before_test_input)
                prompts.append(encapsulate_prompt(prompt))

            print("-" * 60)
            print(prompts[1])
            print("-" * 60)

            for prompt in tqdm(prompts, ncols=60):
                response = client.chat.completions.create(
                    messages=prompt,
                    model=model_name,
                    max_tokens=max_new_tokens
                )

                output = response.choices[0].message.content
                output = dataset.parse_output(output, is_cot=is_cot, gen_text=prompt[0]["content"]+"\n"+output.strip())
                
                if isinstance(output, tuple):
                    output, new_prompt = output
                    new_prompt = encapsulate_prompt(new_prompt)
                    response = client.chat.completions.create(
                        messages=new_prompt,
                        model=model_name,
                        max_tokens=max_new_tokens
                    )
                    new_output = response.choices[0].message.content
                    output = new_output.replace("\n", " ")

                f.write(output + "\n")
        
        print("=====================================")
        print(f"Model: {model_name}")
        print(f"Task: {dataset_name}")
        print(f"Shot: {shot}")
        print(f"Template: {template}")
        print(f"Using CoT: {is_cot}")
        print("=====================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs='+', required=True, help="Name of the dataset. Can be multiple datasets.")
    parser.add_argument("--shot", nargs='+', default=["4"], help="Number of examples to use for each test input")
    parser.add_argument("-t", "--templates", nargs='+', default=["vanilla", "ensemble_random"], help="Templates to use for generation")
    parser.add_argument("--cot_mode", type=int, default=2, help="Whether to generate COT prompts. 0 for vanilla, 1 for COT, 2 for both")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--num_nouns", type=int, default=2, help="Number of random nouns to use for ensemble templates")
    parser.add_argument("--models", nargs='+', default=["gpt-3.5-turbo-0125"], help="List of models to use for generation")
    args = parser.parse_args()

    is_cot = {0: False, 1: True, 2: "both"}[args.cot_mode]
    dataset_names = args.dataset
    models = args.models
    shots = [int(shot) for shot in args.shot]
    
    for dataset_name in dataset_names:
        for model in models:
            for shot in shots:
                main(dataset_name, model, shot, args.templates, is_cot, max_new_tokens=args.max_new_tokens, num_nouns=args.num_nouns)
