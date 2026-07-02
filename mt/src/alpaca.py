import random
import torch
import os 

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

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
            if not (os.path.exists(f"../output/xglm-{i}") or 
                    os.path.exists(f"../output/alpaca-{i}") or 
                    os.path.exists(f"../result/xglm-{i}") or 
                    os.path.exists(f"../result/alpaca-{i}")):
                suffix = f"-{i}"
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
        if (os.path.exists(f"../output/xglm-{i}") or 
            os.path.exists(f"../output/alpaca-{i}") or 
            os.path.exists(f"../result/xglm-{i}") or 
            os.path.exists(f"../result/alpaca-{i}")):
            latest_suffix = f"-{i}"
            i += 1
        else:
            break
    return latest_suffix

def lang_map(lang):
    lang_dict = {'en': 'English', 'de': 'German', 'fr': 'French', 'ru': 'Russian'}
    return lang_dict[lang]

noun_list = []
noun_cnt = 0

def init_noun():
    global noun_list, noun_cnt
    with open("noun_65536.txt", "r") as f:
        for line in f:
            noun_list.append(line.strip())
    noun_cnt = 0


def get_next_noun():
    global noun_list, noun_cnt
    if noun_cnt == len(noun_list):
        noun_cnt = 0
    noun_cnt += 1
    return noun_list[noun_cnt - 1]


def idx2example(train_sentence_pairs, idxs, shot, order):
    THRESHOLD = 120
    final_idxs = []
    final_examples = []
    count, i = 0, 0

    while count < shot and i < len(idxs):
        idx = idxs[i]
        src = train_sentence_pairs[idx][0].strip('"').split()
        if len(src) > THRESHOLD:
            i += 1
            continue
        final_idxs.append(idx)
        i += 1
        count += 1
    
    while count < shot:
        idx = random.randint(0, len(train_sentence_pairs) - 1)
        src = train_sentence_pairs[idx][0].strip('"').split()
        if len(src) > THRESHOLD:
            continue
        final_idxs.append(idx)
        count += 1

    for idx in final_idxs:
        final_examples.append(train_sentence_pairs[idx])

    final_examples = do_order(final_examples, order)

    return final_examples


def get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template="alpaca"):
    if template == "alpaca_ensemble":
        return template_alpaca_ensemble(test_sentence, train_pairs_list, src_lang, tgt_lang)
    elif template == "alpaca_ensemble_testinst":
        return template_alpaca_ensemble_testinst(test_sentence, train_pairs_list, src_lang, tgt_lang)
    elif template == "ensemble_word_syntax":
        return template_ensemble_word_syntax(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_word_syntax_different":
        return template_ensemble_word_syntax_different(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_syntax_word":
        return template_ensemble_syntax_word(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_word_semantics":
        return template_ensemble_word_semantics(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_random_random":
        return template_ensemble_random_random(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "single_random":
        return template_single_random(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "single_example":
        return template_single_example(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    else:
        raise ValueError(f'Invalid template: {template}')


def template_alpaca_ensemble(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    instruction = f'Instruction: Translate the following {src_lang} text into {tgt_lang}.'
    prompt.append(instruction)

    for train_pairs in train_pairs_list:
        for train_pair in train_pairs:
            src_prompt = f'{src_lang}: {train_pair[0]}'
            tgt_prompt = f'{tgt_lang}: {train_pair[1]}'
            prompt.append(src_prompt)
            prompt.append(tgt_prompt)
    
    test_input = f'{src_lang}: {test_sentence}'
    tgt_prompt = f'{tgt_lang}:'
    prompt.append(test_input)
    prompt.append(tgt_prompt)

    prompt = "\n".join(prompt)

    return prompt


def template_alpaca_ensemble_testinst(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    instruction = f'Instruction: Translate the following {src_lang} text into {tgt_lang}.'
    prompt.append(instruction)

    for train_pairs in train_pairs_list:
        for train_pair in train_pairs:
            src_prompt = f'{src_lang}: {train_pair[0]}'
            tgt_prompt = f'{tgt_lang}: {train_pair[1]}'
            prompt.append(src_prompt)
            prompt.append(tgt_prompt)
    
    prompt.append(f'Translate the following sentence:')
    test_input = f'{src_lang}: {test_sentence}'
    tgt_prompt = f'{tgt_lang}:'
    prompt.append(test_input)
    prompt.append(tgt_prompt)

    prompt = "\n".join(prompt)

    return prompt


def template_ensemble_word_syntax(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f'Instruction: Translate {src_lang} text into {tgt_lang}.')
    prompt.append(f'Examples with similar words:')
    for train_pair in train_pairs_list[0]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Examples with similar syntax:')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def template_ensemble_word_syntax_different(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f'Instruction: Translate {src_lang} text into {tgt_lang}.')
    prompt.append(f'Examples with different words:')
    for train_pair in train_pairs_list[0]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Examples with different syntax:')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def template_ensemble_syntax_word(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f'Instruction: Translate {src_lang} text into {tgt_lang}.')
    prompt.append(f'Examples with similar syntax:')
    for train_pair in train_pairs_list[0]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Examples with similar words:')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def template_ensemble_word_semantics(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f'Instruction: Translate {src_lang} text into {tgt_lang}.')
    prompt.append(f'Examples with similar words:')
    for train_pair in train_pairs_list[0]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Examples with similar semantics:')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def template_ensemble_random_random(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f'Instruction: Translate {src_lang} text into {tgt_lang}.')

    n = get_next_noun()
    prompt.append(f'Examples with similar {n}:')
    for train_pair in train_pairs_list[0]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')

    n = get_next_noun()
    prompt.append(f'Examples with similar {n}:')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def template_single_random(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f'Instruction: Translate {src_lang} text into {tgt_lang}.')

    n = get_next_noun()
    prompt.append(f'Examples with similar {n}:')
    for train_pair in train_pairs_list[0]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def template_single_example(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f'Instruction: Translate {src_lang} text into {tgt_lang}.')

    prompt.append(f'Examples:')
    for train_pair in train_pairs_list[0]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def extract_answer(generated_text):
    if '\n' in generated_text:
        generated_text = generated_text.split('\n')[0]
    generated_text = generated_text.strip()
    return generated_text


def read_idx_file(fn):
    idx_list = []
    with open(fn, "r") as f:
        for line in f:
            line = line.strip()
            idxs = line.split(" ")
            idxs = [int(idx) for idx in idxs]
            idx_list.append(idxs)
    return idx_list


def do_order(list, order):
    if order == "descending":
        return list
    elif order == "ascending":
        return list[::-1]
    elif order == "random":
        return random.shuffle(list)


def main(selections=["bm25"], order="descending", langs=["de", "fr", "ru"], directions=["into", "outof"], output_dir="../output/alpaca", shot=4, batch_size=4, templates=["a"], cut=-1):
    model_name_or_path = 'wxjiao/alpaca-7b'
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    gen_config = GenerationConfig(
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token=tokenizer.pad_token_id,
                )

    for template in templates:
        for direction in directions:
            for lang in langs:
                for selection in selections:
                    if 'random' in template:
                        init_noun()

                    if direction == "into":
                        src_lang = lang
                        tgt_lang = "en"
                        test_src_fn = f"../data/{lang}/test.{lang}"
                        test_tgt_fn = f"../data/{lang}/test.en"
                        train_src_fn = f"../data/{lang}/train.{lang}"
                        train_tgt_fn = f"../data/{lang}/train.en"
                        idx_dir = f"../data/{lang}/index/test/into"
                    else:
                        src_lang = "en"
                        tgt_lang = lang
                        test_src_fn = f"../data/{lang}/test.en"
                        test_tgt_fn = f"../data/{lang}/test.{lang}"
                        train_src_fn = f"../data/{lang}/train.en"
                        train_tgt_fn = f"../data/{lang}/train.{lang}"
                        idx_dir = f"../data/{lang}/index/test/outof"

                    selection_list = selection.split("+")
                    idx_list_list = []
                    for sel in selection_list:
                        idx_fn = f"{idx_dir}/{sel}.index"
                        idx_list = read_idx_file(idx_fn)
                        idx_list_list.append(idx_list)

                    test_sentences = []
                    with open(test_src_fn, "r") as f:
                        for line in f:
                            test_sentences.append(line.strip())
                    gold = []
                    with open(test_tgt_fn, "r") as f:
                        for line in f:
                            gold.append(line.strip())
                    if cut > 0:
                        test_sentences = test_sentences[:cut]
                        gold = gold[:cut]
                    
                    train_sentence_pairs = []
                    with open(train_src_fn, "r") as f1, open(train_tgt_fn, "r") as f2:
                        for src, tgt in zip(f1, f2):
                            train_sentence_pairs.append((src.strip(), tgt.strip()))

                    suffix = get_run_suffix(is_inference=True)
                    if output_dir.endswith("alpaca"):
                        curr_output_dir = output_dir + suffix
                    else:
                        curr_output_dir = os.path.join(output_dir, f"alpaca{suffix}")
                    output_fn = f"{curr_output_dir}/{lang}.{direction}.{selection}.{shot}.{order}.{template}.txt"
                    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

                    with open(output_fn, "w") as f:
                        prompts = []
                        for i in tqdm(range(len(test_sentences)), ncols=60):
                            test_sentence = test_sentences[i]
                            train_pairs_list = []
                            count_dict = {}
                            for selection in selection_list:
                                count_dict[selection] = 0
                            for j in range(len(selection_list)):
                                idx_list = idx_list_list[j]
                                selection = selection_list[j]
                                train_pairs = idx2example(train_sentence_pairs, idx_list[i][count_dict[selection] * (shot // len(idx_list_list)):], shot // len(idx_list_list), order)
                                train_pairs_list.append(train_pairs)
                                count_dict[selection] += 1
                            prompt = get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template=template)
                            prompts.append(prompt)

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
                                new_text = gen_text.replace(original_text, "").split('\n')[0].strip()
                                f.write(new_text + "\n")
                    
                    print("=====================================")
                    print(f"Language: {lang}")
                    print(f"Direction: {direction}")
                    print(f"Selection: {selection}")
                    print(f"Shot: {shot}")
                    print(f"Order: {order}")
                    print(f"Template: {template}")
                    print("=====================================")


if __name__ == "__main__":
    selections = ["rand1+rand1", "bm25+polynomial", "polynomial+bm25"]
    order = "ascending"
    langs = ["de", "fr", "ru"]
    directions = ["into"]
    output_dir = "../output/alpaca"
    shot = 4
    batch_size = 8
    templates = ["alpaca_ensemble", "ensemble_word_syntax", "ensemble_random_random"]
    cut = -1
    main(selections, order, langs, directions, output_dir, shot, batch_size, templates, cut)