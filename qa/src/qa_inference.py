import os
from tqdm import tqdm
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from data_loader import get_dataset, ReasoningData
from template import get_prompt_generator

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Replace with the path to the file containing random nouns
PATH_TO_RAND_NOUNS = "./noun/noun_65536.txt"

def main(
        dataset_name, 
        model_name_or_path="wxjiao/alpaca-7b", 
        shot=4, 
        batch_size=16, 
        templates=["vanilla", "ensemble_random"], 
        is_cot: Union[bool, str]=False,
        max_new_tokens=256,
        num_nouns=2,
        **kwargs
        ):
    """
    - is_cot: Whether to generate COT prompts. Default is False. Set "both" to generate both vanilla and COT prompts.
    """
    if is_cot == "both":
        main(dataset_name, model_name_or_path, shot, batch_size, templates, True, max_new_tokens, num_nouns, **kwargs)
        is_cot = False

    if isinstance(templates, str):
        templates = [templates]

    dataset: ReasoningData = get_dataset(dataset_name, **kwargs)
    output_dir = dataset.data_dir.replace("data", "output")
    model_name = model_name_or_path.split("/")[-1]
    output_dir = f"{output_dir}/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_exist = True
    for template in templates:
        output_fn = f"{output_dir}/{shot}.{template}.txt" if not is_cot \
        else f"{output_dir}/{shot}.{template}.cot.txt"
        if not (os.path.exists(output_fn) and os.path.getsize(output_fn) > 0):
            all_exist = False
            break
    if all_exist:
        print(f"All outputs for {model_name} on {dataset_name} (shot {shot}, CoT {is_cot}) already exist. Skipping model loading and generation.")
        return

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    gen_config = GenerationConfig(
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token=tokenizer.pad_token_id,
                )
                
    examples, (test_inputs, gold_answers) = dataset.load_data(is_cot=is_cot, shot=shot)

    for template in templates:
        output_fn = f"{output_dir}/{shot}.{template}.txt" if not is_cot \
        else f"{output_dir}/{shot}.{template}.cot.txt"

        if os.path.exists(output_fn) and os.path.getsize(output_fn) > 0:
            print(f"Output file {output_fn} already exists. Skipping this template.")
            continue

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
                prompts.append(prompt)

            print("-" * 60)
            print(prompts[1])
            print("-" * 60)

            for i in tqdm(range(0, len(prompts), batch_size), ncols=60):
                p = prompts[i:i+batch_size]
                tokenized = tokenizer(p, padding=True, return_tensors="pt")
                input_ids = tokenized.input_ids.cuda()
                attn_mask = tokenized.attention_mask.cuda()
                input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
                attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

                with torch.no_grad():
                    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id)

                for original_input, gen_id in zip(input_ids, generated_ids):
                    original_text = tokenizer.decode(original_input, skip_special_tokens=True)
                    gen_text = tokenizer.decode(gen_id, skip_special_tokens=True)
                    new_text = gen_text.replace(original_text, "")
                    
                    output = dataset.parse_output(new_text, gen_text=gen_text, is_cot=is_cot)
                    if isinstance(output, tuple):
                        output, new_prompt = output
                        new_tokenized = tokenizer(new_prompt, return_tensors="pt")
                        new_input_ids = new_tokenized.input_ids.cuda()
                        new_attn_mask = new_tokenized.attention_mask.cuda()
                        new_text = model.generate(inputs=new_input_ids, attention_mask=new_attn_mask, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id)
                        new_text = tokenizer.decode(new_text[0], skip_special_tokens=True)
                        new_text = new_text.replace(new_prompt, "").strip()
                        output = new_text.replace("\n", " ")

                    f.write(output + "\n")
        
        print("=====================================")
        print(f"Model: {model_name}")
        print(f"Task: {dataset_name}")
        print(f"Shot: {shot}")
        print(f"Template: {template}")
        print(f"Using CoT: {is_cot}")
        print("=====================================")

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="0", help="GPU device")
    parser.add_argument("--dataset", nargs='+', required=True, help="Name of the dataset. Can be multiple datasets.")
    parser.add_argument("--shot", nargs='+', default=["4"], help="Number of examples to use for each test input")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("-t", "--templates", nargs='+', default=["vanilla", "ensemble_random"], help="Templates to use for generation")
    parser.add_argument("--cot_mode", type=int, default=2, help="Whether to generate COT prompts. 0 for vanilla, 1 for COT, 2 for both")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--num_nouns", type=int, default=2, help="Number of random nouns to use for ensemble templates")
    parser.add_argument("--models", nargs='+', default=["alpaca", "llama2", "mistral"], help="List of models to use for generation")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    is_cot = {0: False, 1: True, 2: "both"}[args.cot_mode]
    dataset_names = args.dataset
    model_name_map = {
        "alpaca": "wxjiao/alpaca-7b",
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B"
    }
    models = [model_name_map[model] for model in args.models]
    shots = [int(shot) for shot in args.shot]
    
    for dataset_name in dataset_names:
        for model in models:
            for shot in shots:
                if shot == 2:
                    bsz = 48
                elif shot == 8:
                    bsz = 12
                elif shot == 16:
                    bsz = 4
                else:
                    bsz = args.batch_size
                main(dataset_name, model, shot, bsz, args.templates, is_cot, max_new_tokens=args.max_new_tokens, num_nouns=args.num_nouns)
