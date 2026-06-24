import math
import random
import torch

from tqdm import tqdm
from transformers import XGLMTokenizer, XGLMForCausalLM
from torch.utils.data import Dataset


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


class MTDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return self.prompts[i]


def lang_map(lang):
    lang_dict = {'en': 'English', 'de': 'German', 'fr': 'French', 'ru': 'Russian'}
    return lang_dict[lang]


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


def get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template="a"):
    if template == "vanilla":
        return template_vanilla(test_sentence, train_pairs_list, src_lang, tgt_lang)
    elif template == "ensemble_word_syntax":
        return template_ensemble_word_syntax(test_sentence, train_pairs_list, src_lang, tgt_lang)
    elif template == "ensemble_syntax_word":
        return template_ensemble_syntax_word(test_sentence, train_pairs_list, src_lang, tgt_lang)
    elif template == "diff_ensemble_word_syntax":
        return template_diff_ensemble_word_syntax(test_sentence, train_pairs_list, src_lang, tgt_lang)
    elif template == "ensemble_word_semantics":
        return template_ensemble_word_semantics(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_random_random":
        return template_ensemble_random_random(test_sentence, train_pairs_list, src_lang, tgt_lang)
    else:
        raise ValueError("Invalid template")


def template_vanilla(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f"Translate from {src_lang} to {tgt_lang}.")
    for train_pair in train_pairs_list[0]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    for train_pair in train_pairs_list[1]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"{test_sentence} = ")
    prompt = "\n###\n".join(prompt)
    return prompt


def template_ensemble_word_syntax(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f"Translate from {src_lang} to {tgt_lang}.")
    prompt.append(f"Examples with similar words:")
    for train_pair in train_pairs_list[0]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Examples with similar syntax:")
    for train_pair in train_pairs_list[1]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Translate the following sentence:")
    prompt.append(f"{test_sentence} = ")
    prompt = "\n###\n".join(prompt)
    return prompt


def template_ensemble_syntax_word(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f"Translate from {src_lang} to {tgt_lang}.")
    prompt.append(f"Examples with similar syntax:")
    for train_pair in train_pairs_list[0]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Examples with similar words:")
    for train_pair in train_pairs_list[1]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Translate the following sentence:")
    prompt.append(f"{test_sentence} = ")
    prompt = "\n###\n".join(prompt)
    return prompt


def template_diff_ensemble_word_syntax(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f"Translate from {src_lang} to {tgt_lang}.")
    prompt.append(f"Examples with different words:")
    for train_pair in train_pairs_list[0]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Examples with different syntax:")
    for train_pair in train_pairs_list[1]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Translate the following sentence:")
    prompt.append(f"{test_sentence} = ")
    prompt = "\n###\n".join(prompt)
    return prompt


def template_ensemble_word_semantics(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f"Translate from {src_lang} to {tgt_lang}.")
    prompt.append(f"Examples with similar words:")
    for train_pair in train_pairs_list[0]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Examples with similar semantics:")
    for train_pair in train_pairs_list[1]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Translate the following sentence:")
    prompt.append(f"{test_sentence} = ")
    prompt = "\n###\n".join(prompt)
    return prompt


def template_ensemble_random_random(test_sentence, train_pairs_list, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    prompt.append(f"Translate from {src_lang} to {tgt_lang}.")
    noun = get_next_noun()
    prompt.append(f"Examples with similar {noun}:")
    for train_pair in train_pairs_list[0]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    noun = get_next_noun()
    prompt.append(f"Examples with similar {noun}:")
    for train_pair in train_pairs_list[1]:
        prompt.append(f"{train_pair[0]} = {train_pair[1]}")
    prompt.append(f"Translate the following sentence:")
    prompt.append(f"{test_sentence} = ")
    prompt = "\n###\n".join(prompt)
    return prompt


def extract_attention(input_ids, attention_matrix, tokenizer, shot, template="a"):
    if template == "ensemble_word_syntax":
        return extract_attention_ensemble_word_syntax(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "diff_ensemble_word_syntax":
        return extract_attention_diff_ensemble_word_syntax(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_syntax_word":
        return extract_attention_ensemble_syntax_word(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_word_semantics":
        return extract_attention_ensemble_word_semantics(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_random_random":
        return extract_attention_ensemble_random_random(input_ids, attention_matrix, tokenizer, shot=shot)
    else:
        raise NotImplementedError
    

def extract_attention_ensemble_word_syntax(input_ids, attention_matrix, tokenizer, shot=4):
    double_sharp_id = 85690
    double_sharp_poss = [i for i, x in enumerate(input_ids.tolist()) if x == double_sharp_id]

    word_pos = double_sharp_poss[0] + 5
    word_examples_poss_list = []
    for i in range(shot // 2):
        word_example_poss = range(double_sharp_poss[1 + i] + 1, double_sharp_poss[1 + i + 1] - 1)
        word_examples_poss_list.append(word_example_poss)
    
    syntax_pos = double_sharp_poss[2 * (shot // 2) - 1] + 5
    syntax_examples_poss_list = []
    for i in range(shot // 2):
        syntax_example_poss = range(double_sharp_poss[2 * (shot // 2) + i] + 1, double_sharp_poss[2 * (shot // 2) + i + 1] - 1)
        syntax_examples_poss_list.append(syntax_example_poss)
    
    last_pos = len(input_ids) - 1

    worde2word = []
    syntaxe2word = []
    syntaxe2syntax = []
    pred2word = 0.0
    pred2syntax = 0.0

    for i in range(shot // 2):
        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        worde2word.append(attention_word_example_and_word)

        syntax_example_poss = syntax_examples_poss_list[i]
        attention_syntax_example_and_word = attention_matrix[syntax_example_poss, word_pos].mean().item()
        attention_syntax_example_and_syntax = attention_matrix[syntax_example_poss, syntax_pos].mean().item()
        syntaxe2word.append(attention_syntax_example_and_word)
        syntaxe2syntax.append(attention_syntax_example_and_syntax)

    worde2word = sum(worde2word) / len(worde2word)
    syntaxe2word = sum(syntaxe2word) / len(syntaxe2word)
    syntaxe2syntax = sum(syntaxe2syntax) / len(syntaxe2syntax)
    
    pred2word = attention_matrix[last_pos, word_pos].item()
    pred2syntax = attention_matrix[last_pos, syntax_pos].item()

    results = {
        "worde2word": worde2word,
        "syntaxe2word": syntaxe2word,
        "syntaxe2syntax": syntaxe2syntax,
        "pred2word": pred2word,
        "pred2syntax": pred2syntax,
    }
    
    return results


def extract_attention_diff_ensemble_word_syntax(input_ids, attention_matrix, tokenizer, shot=4):
    double_sharp_id = 85690
    double_sharp_poss = [i for i, x in enumerate(input_ids.tolist()) if x == double_sharp_id]

    word_pos = double_sharp_poss[0] + 5
    word_examples_poss_list = []
    for i in range(shot // 2):
        word_example_poss = range(double_sharp_poss[1 + i] + 1, double_sharp_poss[1 + i + 1] - 1)
        word_examples_poss_list.append(word_example_poss)
    
    syntax_pos = double_sharp_poss[2 * (shot // 2) - 1] + 5
    syntax_examples_poss_list = []
    for i in range(shot // 2):
        syntax_example_poss = range(double_sharp_poss[2 * (shot // 2) + i] + 1, double_sharp_poss[2 * (shot // 2) + i + 1] - 1)
        syntax_examples_poss_list.append(syntax_example_poss)
    
    last_pos = len(input_ids) - 1

    worde2word = []
    syntaxe2word = []
    syntaxe2syntax = []
    pred2word = 0.0
    pred2syntax = 0.0

    for i in range(shot // 2):
        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        worde2word.append(attention_word_example_and_word)

        syntax_example_poss = syntax_examples_poss_list[i]
        attention_syntax_example_and_word = attention_matrix[syntax_example_poss, word_pos].mean().item()
        attention_syntax_example_and_syntax = attention_matrix[syntax_example_poss, syntax_pos].mean().item()
        syntaxe2word.append(attention_syntax_example_and_word)
        syntaxe2syntax.append(attention_syntax_example_and_syntax)

    worde2word = sum(worde2word) / len(worde2word)
    syntaxe2word = sum(syntaxe2word) / len(syntaxe2word)
    syntaxe2syntax = sum(syntaxe2syntax) / len(syntaxe2syntax)
    
    pred2word = attention_matrix[last_pos, word_pos].item()
    pred2syntax = attention_matrix[last_pos, syntax_pos].item()

    results = {
        "worde2word": worde2word,
        "syntaxe2word": syntaxe2word,
        "syntaxe2syntax": syntaxe2syntax,
        "pred2word": pred2word,
        "pred2syntax": pred2syntax,
    }
    
    return results


def extract_attention_ensemble_syntax_word(input_ids, attention_matrix, tokenizer, shot=4):
    double_sharp_id = 85690
    double_sharp_poss = [i for i, x in enumerate(input_ids.tolist()) if x == double_sharp_id]

    syntax_pos = double_sharp_poss[0] + 5
    syntax_examples_poss_list = []
    for i in range(shot // 2):
        syntax_example_poss = range(double_sharp_poss[1 + i] + 1, double_sharp_poss[1 + i + 1] - 1)
        syntax_examples_poss_list.append(syntax_example_poss)
    
    word_pos = double_sharp_poss[2 * (shot // 2) - 1] + 5
    word_examples_poss_list = []
    for i in range(shot // 2):
        word_example_poss = range(double_sharp_poss[2 * (shot // 2) + i] + 1, double_sharp_poss[2 * (shot // 2) + i + 1] - 1)
        word_examples_poss_list.append(word_example_poss)
    
    last_pos = len(input_ids) - 1

    syntaxe2syntax = []
    worde2syntax = []
    worde2word = []
    pred2syntax = 0.0
    pred2word = 0.0

    for i in range(shot // 2):
        syntax_example_poss = syntax_examples_poss_list[i]
        attention_syntax_example_and_syntax = attention_matrix[syntax_example_poss, syntax_pos].mean().item()
        syntaxe2syntax.append(attention_syntax_example_and_syntax)

        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_syntax = attention_matrix[word_example_poss, syntax_pos].mean().item()
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        worde2syntax.append(attention_word_example_and_syntax)
        worde2word.append(attention_word_example_and_word)

    syntaxe2syntax = sum(syntaxe2syntax) / len(syntaxe2syntax)
    worde2syntax = sum(worde2syntax) / len(worde2syntax)
    worde2word = sum(worde2word) / len(worde2word)
    
    pred2syntax = attention_matrix[last_pos, syntax_pos].item()
    pred2word = attention_matrix[last_pos, word_pos].item()

    results = {
        "syntaxe2syntax": syntaxe2syntax,
        "worde2syntax": worde2syntax,
        "worde2word": worde2word,
        "pred2syntax": pred2syntax,
        "pred2word": pred2word,
    }
    
    return results


def extract_attention_ensemble_word_semantics(input_ids, attention_matrix, tokenizer, shot=4):
    double_sharp_id = 85690
    double_sharp_poss = [i for i, x in enumerate(input_ids.tolist()) if x == double_sharp_id]

    word_pos = double_sharp_poss[0] + 5
    word_examples_poss_list = []
    for i in range(shot // 2):
        word_example_poss = range(double_sharp_poss[1 + i] + 1, double_sharp_poss[1 + i + 1] - 1)
        word_examples_poss_list.append(word_example_poss)
    
    semantics_pos = double_sharp_poss[2 * (shot // 2) - 1] + 5
    semantics_examples_poss_list = []
    for i in range(shot // 2):
        semantics_example_poss = range(double_sharp_poss[2 * (shot // 2) + i] + 1, double_sharp_poss[2 * (shot // 2) + i + 1] - 1)
        semantics_examples_poss_list.append(semantics_example_poss)
    
    last_pos = len(input_ids) - 1

    worde2word = []
    semanticse2word = []
    semanticse2semantics = []
    pred2word = 0.0
    pred2semantics = 0.0

    for i in range(shot // 2):
        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        worde2word.append(attention_word_example_and_word)

        semantics_example_poss = semantics_examples_poss_list[i]
        attention_semantics_example_and_word = attention_matrix[semantics_example_poss, word_pos].mean().item()
        attention_semantics_example_and_semantics = attention_matrix[semantics_example_poss, semantics_pos].mean().item()
        semanticse2word.append(attention_semantics_example_and_word)
        semanticse2semantics.append(attention_semantics_example_and_semantics)

    worde2word = sum(worde2word) / len(worde2word)
    semanticse2word = sum(semanticse2word) / len(semanticse2word)
    semanticse2semantics = sum(semanticse2semantics) / len(semanticse2semantics)
    
    pred2word = attention_matrix[last_pos, word_pos].item()
    pred2semantics = attention_matrix[last_pos, semantics_pos].item()

    results = {
        "worde2word": worde2word,
        "semanticse2word": semanticse2word,
        "semanticse2semantics": semanticse2semantics,
        "pred2word": pred2word,
        "pred2semantics": pred2semantics,
    }
    
    return results


def extract_attention_ensemble_random_random(input_ids, attention_matrix, tokenizer, shot=4):
    double_sharp_id = 85690
    double_sharp_poss = [i for i, x in enumerate(input_ids.tolist()) if x == double_sharp_id]

    r1_pos = double_sharp_poss[0] + 5
    r1_examples_poss_list = []
    for i in range(shot // 2):
        r1_example_poss = range(double_sharp_poss[1 + i] + 1, double_sharp_poss[1 + i + 1] - 1)
        r1_examples_poss_list.append(r1_example_poss)
    
    r2_pos = double_sharp_poss[2 * (shot // 2) - 1] + 5
    r2_examples_poss_list = []
    for i in range(shot // 2):
        r2_example_poss = range(double_sharp_poss[2 * (shot // 2) + i] + 1, double_sharp_poss[2 * (shot // 2) + i + 1] - 1)
        r2_examples_poss_list.append(r2_example_poss)
    
    last_pos = len(input_ids) - 1

    r1e2r1 = []
    r2e2r1 = []
    r2e2r2 = []
    pred2r1 = 0.0
    pred2r2 = 0.0

    for i in range(shot // 2):
        r1_example_poss = r1_examples_poss_list[i]
        attention_r1_example_and_r1 = attention_matrix[r1_example_poss, r1_pos].mean().item()
        r1e2r1.append(attention_r1_example_and_r1)

        r2_example_poss = r2_examples_poss_list[i]
        attention_r2_example_and_r1 = attention_matrix[r2_example_poss, r1_pos].mean().item()
        attention_r2_example_and_r2 = attention_matrix[r2_example_poss, r2_pos].mean().item()
        r2e2r1.append(attention_r2_example_and_r1)
        r2e2r2.append(attention_r2_example_and_r2)
    
    r1e2r1 = sum(r1e2r1) / len(r1e2r1)
    r2e2r1 = sum(r2e2r1) / len(r2e2r1)
    r2e2r2 = sum(r2e2r2) / len(r2e2r2)
    
    pred2r1 = attention_matrix[last_pos, r1_pos].item()
    pred2r2 = attention_matrix[last_pos, r2_pos].item()

    results = {
        "r1e2r1": r1e2r1,
        "r2e2r1": r2e2r1,
        "r2e2r2": r2e2r2,
        "pred2r1": pred2r1,
        "pred2r2": pred2r2,
    }
    
    return results


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


def main(device="cuda:0", selections=["bm25-polynomial"], order="descending", langs=["de", "fr", "ru"], directions=["into", "outof"], model_path="facebook/xglm-7.5B", shot=4, templates=["a"], cut=100, output_path="../result/xglm/attention.tsv"):
    torch.device(device)
    tokenizer = XGLMTokenizer.from_pretrained(model_path, padding_side="left")
    model = XGLMForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

    for template in templates:
        for selection in selections:
            results_sum_overalls = []
            for direction in directions:
                for lang in langs:
                    if "random" in template:
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

                    prompts = []
                    for i in range(len(test_sentences)):
                        test_sentence = test_sentences[i]
                        train_pairs_list = []
                        count_dict = {}
                        for sel in selection_list:
                            count_dict[sel] = 0
                        for j in range(len(selection_list)):
                            idx_list = idx_list_list[j]
                            sel = selection_list[j]
                            train_pairs = idx2example(train_sentence_pairs, idx_list[i][count_dict[sel]*(shot//len(idx_list_list)):], shot//len(idx_list_list), order)
                            train_pairs_list.append(train_pairs)
                            count_dict[sel] += 1
                        prompt = get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template)
                        prompts.append(prompt)
                    
                    results_sums = []
                    for prompt in tqdm(prompts, ncols=60):
                        input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(device)
                        with torch.no_grad():
                            attentions = model(input_ids, output_attentions=True).attentions
                            # Move attention tensors to CPU immediately to free GPU memory
                            attention_matrix = [layer_attn[0].cpu() for layer_attn in attentions]
                            del attentions
                        input_ids = input_ids[0].cpu()
                        for layer in range(len(attention_matrix)):
                            layer_attention_matrix = attention_matrix[layer]
                            if len(results_sums) <= layer:
                                results_sum = {}
                                results_sums.append(results_sum)
                            else:
                                results_sum = results_sums[layer]
                            layer_attention_matrix = layer_attention_matrix.mean(dim=0)
                            attention_results = extract_attention(input_ids, layer_attention_matrix, tokenizer, shot, template)
                            for key in attention_results:
                                if type(attention_results[key]) == list:
                                    if key not in results_sum:
                                        results_sum[key] = [0.0] * len(attention_results[key])
                                    for k in range(len(attention_results[key])):
                                        if math.isnan(attention_results[key][k]):
                                            attention_results[key][k] = 0.0
                                        results_sum[key][k] += attention_results[key][k]
                                else:
                                    if key not in results_sum:
                                        results_sum[key] = 0.0
                                    if math.isnan(attention_results[key]):
                                        attention_results[key] = 0.0
                                    results_sum[key] += attention_results[key]

                    if results_sum_overalls == []:
                        results_sum_overalls = [{} for _ in range(len(results_sums))]

                    for layer, results_sum in enumerate(results_sums):
                        for key in results_sum:
                            if type(results_sum[key]) == list:
                                for i in range(len(results_sum[key])):
                                    results_sum[key][i] /= len(prompts)
                            else:
                                results_sum[key] /= len(prompts)
                        
                        # add to overall results
                        results_sum_overall = results_sum_overalls[layer]
                        for key in results_sum:
                            if type(results_sum[key]) == list:
                                if key not in results_sum_overall:
                                    results_sum_overall[key] = [0.0] * len(results_sum[key])
                                for i in range(len(results_sum[key])):
                                    results_sum_overall[key][i] += results_sum[key][i]
                            else:
                                if key not in results_sum_overall:
                                    results_sum_overall[key] = 0.0
                                results_sum_overall[key] += results_sum[key]

            # average over all directions and languages
            for results_sum_overall in results_sum_overalls:
                for key in results_sum_overall:
                    if type(results_sum_overall[key]) == list:
                        for i in range(len(results_sum_overall[key])):
                            results_sum_overall[key][i] /= len(langs) * len(directions)
                    else:
                        results_sum_overall[key] /= len(langs) * len(directions)


            print("=====================================")
            print(f"Selection: {selection}")
            print(f"Shot: {shot}")
            print(f"Order: {order}")
            print(f"Template: {template}")
            # print('-------------------------------------')
            # for key in results_sum_overalls[0]:
            #     if type(results_sum_overalls[0][key]) == list:
            #         for i in range(len(results_sum_overalls[0][key])):
            #             print(f"{key}[{i}]: {10000*results_sum_overalls[0][key][i]:.2f}")
            #     else:
            #         print(f"{key}: {10000*results_sum_overalls[0][key]:.2f}")
            print("=====================================")

            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "a") as f:
                for results_sum_overall in results_sum_overalls:
                    f.write(f"{selection}\t{shot}\t{order}\t{template}\t")
                    for key in results_sum_overall:
                        if type(results_sum_overall[key]) == list:
                            for i in range(len(results_sum_overall[key])):
                                f.write(f"{10000*results_sum_overall[key][i]:.2f}\t")
                        else:
                            f.write(f"{10000*results_sum_overall[key]:.2f}\t")
                    f.write("\n")



if __name__ == "__main__":
    device = "cuda"
    selections = ["bm25+polynomial"]
    order = "ascending"
    langs = ["de", "fr", "ru"]
    directions = ["into", "outof"]
    model_path = "facebook/xglm-7.5B"
    shot = 4
    templates = ["ensemble_word_syntax", "ensemble_random_random"]
    cut = -1
    main(device, selections, order, langs, directions, model_path, shot, templates, cut)