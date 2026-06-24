import random
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template="alpaca", tokenizer=None):
    if template == "alpaca_ensemble":
        return template_alpaca_ensemble(test_sentence, train_pairs_list, src_lang, tgt_lang)
    elif template == "ensemble_word_syntax":
        return template_ensemble_word_syntax(test_sentence, train_pairs_list[:2], src_lang, tgt_lang, tokenizer=tokenizer)
    elif template == "ensemble_word_syntax_different":
        return template_ensemble_word_syntax_different(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_syntax_word":
        return template_ensemble_syntax_word(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_word_semantics":
        return template_ensemble_word_semantics(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
    elif template == "ensemble_random_random":
        return template_ensemble_random_random(test_sentence, train_pairs_list[:2], src_lang, tgt_lang)
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


def template_ensemble_word_syntax(test_sentence, train_pairs_list, src_lang, tgt_lang, tokenizer=None):
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
    with open("noun.txt", "a") as f:
        f.write(n + "\n")

    n = get_next_noun()
    prompt.append(f'Examples with similar {n}:')
    for train_pair in train_pairs_list[1]:
        prompt.append(f'{src_lang}: {train_pair[0]}')
        prompt.append(f'{tgt_lang}: {train_pair[1]}')
    with open("noun.txt", "a") as f:
        f.write(n + "\n")
    prompt.append(f'Translate the following sentence:')
    prompt.append(f'{src_lang}: {test_sentence}')
    prompt.append(f'{tgt_lang}:')
    prompt = "\n".join(prompt)
    return prompt


def extract_attention(input_ids, attention_matrix, tokenizer, shot, template="alpaca"):
    if template == "alpaca_ensemble":
        return extract_attention_alpaca_ensemble(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_word_syntax":
        return extract_attention_ensemble_word_syntax(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_word_syntax_different":
        return extract_attention_ensemble_word_syntax_different(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_syntax_word":
        return extract_attention_ensemble_syntax_word(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_word_semantics":
        return extract_attention_ensemble_word_semantics(input_ids, attention_matrix, tokenizer, shot=shot)
    elif template == "ensemble_random_random":
        return extract_attention_ensemble_random_random(input_ids, attention_matrix, tokenizer, shot=shot)
    else:
        raise ValueError(f'Invalid template: {template}')


def extract_attention_alpaca_ensemble(input_ids, attention_matrix, tokenizer, shot):
    newline_id = 13
    newline_poss = [i for i, x in enumerate(input_ids.tolist()) if x == newline_id]

    first_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[i * 2] + 1, newline_poss[i * 2 + 1])
        tgt_poss = range(newline_poss[i * 2 + 1] + 1, newline_poss[i * 2 + 2])
        word_example_poss  = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        first_examples_poss_list.append(word_example_poss)
    
    second_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[2 * (shot // 2) + i * 2] + 1, newline_poss[2 * (shot // 2) + i * 2 + 1])
        tgt_poss = range(newline_poss[2 * (shot // 2) + i * 2 + 1] + 1, newline_poss[2 * (shot // 2) + i * 2 + 2])
        syntax_example_poss = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        second_examples_poss_list.append(syntax_example_poss)
    
    last_pos = len(input_ids) - 1

    pred2firste = []
    pred2seconde = []

    for i in range(shot // 2):
        word_example_poss = first_examples_poss_list[i]
        attention_prediction_and_first_example = attention_matrix[last_pos, word_example_poss].mean().item()
        pred2firste.append(attention_prediction_and_first_example)

        syntax_example_poss = second_examples_poss_list[i]
        attention_prediction_and_second_example = attention_matrix[last_pos, syntax_example_poss].mean().item()
        pred2seconde.append(attention_prediction_and_second_example)
    
    results = {
        "pred2firste": pred2firste,
        "pred2seconde": pred2seconde
    }
    
    return results


def extract_attention_ensemble_word_syntax(input_ids, attention_matrix, tokenizer, shot):
    newline_id = 13
    newline_poss = [i for i, x in enumerate(input_ids.tolist()) if x == newline_id]

    word_pos = newline_poss[0] + 5
    similar_1_pos = newline_poss[0] + 4
    word_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[1 + i * 2] + 1, newline_poss[1 + i * 2 + 1])
        tgt_poss = range(newline_poss[1 + i * 2 + 1] + 1, newline_poss[1 + i * 2 + 2])
        word_example_poss  = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        word_examples_poss_list.append(word_example_poss)
    
    syntax_pos = newline_poss[1 + 2 * (shot // 2)] + 5
    similar_2_pos = newline_poss[1 + 2 * (shot // 2)] + 4
    syntax_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 1])
        tgt_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2 + 1] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 2])
        syntax_example_poss = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        syntax_examples_poss_list.append(syntax_example_poss)
    
    last_pos = len(input_ids) - 1

    worde2word = []
    worde2similar_1 = []
    syntaxe2word = []
    syntaxe2syntax = []
    syntaxe2similar_1 = []
    syntaxe2similar_2 = []
    pred2worde = []
    pred2syntaxe = []
    pred2word = 0.0
    pred2syntax = 0.0
    pred2similar_1 = 0.0
    pred2similar_2 = 0.0

    for i in range(shot // 2):
        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        attention_prediction_and_word_example = attention_matrix[last_pos, word_example_poss].mean().item()
        attention_word_example_and_similar_1 = attention_matrix[word_example_poss, similar_1_pos].mean().item()
        worde2word.append(attention_word_example_and_word)
        pred2worde.append(attention_prediction_and_word_example)
        worde2similar_1.append(attention_word_example_and_similar_1)

        syntax_example_poss = syntax_examples_poss_list[i]
        attention_syntax_example_and_word = attention_matrix[syntax_example_poss, word_pos].mean().item()
        attention_syntax_example_and_syntax = attention_matrix[syntax_example_poss, syntax_pos].mean().item()
        attention_prediction_and_syntax_example = attention_matrix[last_pos, syntax_example_poss].mean().item()
        attention_syntax_example_and_similar_1 = attention_matrix[syntax_example_poss, similar_1_pos].mean().item()
        attention_syntax_example_and_similar_2 = attention_matrix[syntax_example_poss, similar_2_pos].mean().item()
        syntaxe2word.append(attention_syntax_example_and_word)
        syntaxe2syntax.append(attention_syntax_example_and_syntax)
        pred2syntaxe.append(attention_prediction_and_syntax_example)
        syntaxe2similar_1.append(attention_syntax_example_and_similar_1)
        syntaxe2similar_2.append(attention_syntax_example_and_similar_2)
    
    worde2word = sum(worde2word) / len(worde2word)
    syntaxe2word = sum(syntaxe2word) / len(syntaxe2word)
    syntaxe2syntax = sum(syntaxe2syntax) / len(syntaxe2syntax)
    pred2word = attention_matrix[last_pos, word_pos].item()
    pred2syntax = attention_matrix[last_pos, syntax_pos].item()
    pred2similar_1 = attention_matrix[last_pos, similar_1_pos].item()
    pred2similar_2 = attention_matrix[last_pos, similar_2_pos].item()

    results = {
        "worde2word": worde2word,
        # "worde2similar_1": worde2similar_1,
        "syntaxe2word": syntaxe2word,
        "syntaxe2syntax": syntaxe2syntax,
        # "syntaxe2similar_1": syntaxe2similar_1,
        # "syntaxe2similar_2": syntaxe2similar_2,
        # "pred2worde": pred2worde,
        # "pred2syntaxe": pred2syntaxe,
        "pred2word": pred2word,
        "pred2syntax": pred2syntax,
        # "pred2similar_1": pred2similar_1,
        # "pred2similar_2": pred2similar_2
    }
    
    return results


def extract_attention_ensemble_word_syntax_different(input_ids, attention_matrix, tokenizer, shot):
    newline_id = 13
    newline_poss = [i for i, x in enumerate(input_ids.tolist()) if x == newline_id]

    word_pos = newline_poss[1] - 2
    diff_1_pos = newline_poss[1] - 3
    word_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[1 + i * 2] + 1, newline_poss[1 + i * 2 + 1])
        tgt_poss = range(newline_poss[1 + i * 2 + 1] + 1, newline_poss[1 + i * 2 + 2])
        word_example_poss  = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        word_examples_poss_list.append(word_example_poss)
    
    syntax_pos = newline_poss[2 + 2 * (shot // 2)] - 2
    diff_2_pos = newline_poss[2 + 2 * (shot // 2)] - 3
    syntax_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 1])
        tgt_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2 + 1] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 2])
        syntax_example_poss = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        syntax_examples_poss_list.append(syntax_example_poss)
    
    last_pos = len(input_ids) - 1

    worde2word = []
    worde2diff_1 = []
    syntaxe2word = []
    syntaxe2syntax = []
    syntaxe2diff_1 = []
    syntaxe2diff_2 = []
    pred2worde = []
    pred2syntaxe = []
    pred2word = 0.0
    pred2syntax = 0.0
    pred2diff_1 = 0.0
    pred2diff_2 = 0.0

    for i in range(shot // 2):
        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        attention_prediction_and_word_example = attention_matrix[last_pos, word_example_poss].mean().item()
        attention_word_example_and_diff_1 = attention_matrix[word_example_poss, diff_1_pos].mean().item()
        worde2word.append(attention_word_example_and_word)
        pred2worde.append(attention_prediction_and_word_example)
        worde2diff_1.append(attention_word_example_and_diff_1)

        syntax_example_poss = syntax_examples_poss_list[i]
        attention_syntax_example_and_word = attention_matrix[syntax_example_poss, word_pos].mean().item()
        attention_syntax_example_and_syntax = attention_matrix[syntax_example_poss, syntax_pos].mean().item()
        attention_prediction_and_syntax_example = attention_matrix[last_pos, syntax_example_poss].mean().item()
        attention_syntax_example_and_diff_1 = attention_matrix[syntax_example_poss, diff_1_pos].mean().item()
        attention_syntax_example_and_diff_2 = attention_matrix[syntax_example_poss, diff_2_pos].mean().item()
        syntaxe2word.append(attention_syntax_example_and_word)
        syntaxe2syntax.append(attention_syntax_example_and_syntax)
        pred2syntaxe.append(attention_prediction_and_syntax_example)
        syntaxe2diff_1.append(attention_syntax_example_and_diff_1)
        syntaxe2diff_2.append(attention_syntax_example_and_diff_2)
    
    pred2word = attention_matrix[last_pos, word_pos].item()
    pred2syntax = attention_matrix[last_pos, syntax_pos].item()
    pred2diff_1 = attention_matrix[last_pos, diff_1_pos].item()
    pred2diff_2 = attention_matrix[last_pos, diff_2_pos].item()

    results = {
        "worde2word": worde2word,
        "worde2diff_1": worde2diff_1,
        "syntaxe2word": syntaxe2word,
        "syntaxe2syntax": syntaxe2syntax,
        "syntaxe2diff_1": syntaxe2diff_1,
        "syntaxe2diff_2": syntaxe2diff_2,
        "pred2worde": pred2worde,
        "pred2syntaxe": pred2syntaxe,
        "pred2word": pred2word,
        "pred2syntax": pred2syntax,
        "pred2diff_1": pred2diff_1,
        "pred2diff_2": pred2diff_2
    }
    
    return results


def extract_attention_ensemble_syntax_word(input_ids, attention_matrix, tokenizer, shot):
    newline_id = 13
    newline_poss = [i for i, x in enumerate(input_ids.tolist()) if x == newline_id]

    syntax_pos = newline_poss[1] - 2
    similar_1_pos = newline_poss[1] - 3
    syntax_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[1 + i * 2] + 1, newline_poss[1 + i * 2 + 1])
        tgt_poss = range(newline_poss[1 + i * 2 + 1] + 1, newline_poss[1 + i * 2 + 2])
        syntax_example_poss  = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        syntax_examples_poss_list.append(syntax_example_poss)
    
    word_pos = newline_poss[2 + 2 * (shot // 2)] - 2
    similar_2_pos = newline_poss[2 + 2 * (shot // 2)] - 3
    word_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 1])
        tgt_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2 + 1] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 2])
        word_example_poss = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        word_examples_poss_list.append(word_example_poss)
    
    last_pos = len(input_ids) - 1

    syntaxe2syntax = []
    syntaxe2similar_1 = []
    worde2word = []
    worde2syntax = []
    worde2similar_1 = []
    word32similar_2 = []
    pred2syntaxe = []
    pred2worde = []
    pred2syntax = 0.0
    pred2word = 0.0
    pred2similar_1 = 0.0
    pred2similar_2 = 0.0

    for i in range(shot // 2):
        syntax_example_poss = syntax_examples_poss_list[i]
        attention_syntax_example_and_syntax = attention_matrix[syntax_example_poss, syntax_pos].mean().item()
        attention_prediction_and_syntax_example = attention_matrix[last_pos, syntax_example_poss].mean().item()
        attention_syntax_example_and_similar_1 = attention_matrix[syntax_example_poss, similar_1_pos].mean().item()
        syntaxe2syntax.append(attention_syntax_example_and_syntax)
        pred2syntaxe.append(attention_prediction_and_syntax_example)
        syntaxe2similar_1.append(attention_syntax_example_and_similar_1)

        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        attention_word_example_and_syntax = attention_matrix[word_example_poss, syntax_pos].mean().item()
        attention_prediction_and_word_example = attention_matrix[last_pos, word_example_poss].mean().item()
        attention_word_example_and_similar_1 = attention_matrix[word_example_poss, similar_1_pos].mean().item()
        attention_word_example_and_similar_2 = attention_matrix[word_example_poss, similar_2_pos].mean().item()
        worde2word.append(attention_word_example_and_word)
        worde2syntax.append(attention_word_example_and_syntax)
        pred2worde.append(attention_prediction_and_word_example)
        worde2similar_1.append(attention_word_example_and_similar_1)
        word32similar_2.append(attention_word_example_and_similar_2)

    pred2syntax = attention_matrix[last_pos, syntax_pos].item()
    pred2word = attention_matrix[last_pos, word_pos].item()
    pred2similar_1 = attention_matrix[last_pos, similar_1_pos].item()
    pred2similar_2 = attention_matrix[last_pos, similar_2_pos].item()

    results = {
        "syntaxe2syntax": syntaxe2syntax,
        "syntaxe2similar_1": syntaxe2similar_1,
        "worde2word": worde2word,
        "worde2syntax": worde2syntax,
        "worde2similar_1": worde2similar_1,
        "worde2similar_2": word32similar_2,    
        "pred2syntaxe": pred2syntaxe,
        "pred2worde": pred2worde,
        "pred2syntax": pred2syntax,
        "pred2word": pred2word,
        "pred2similar_1": pred2similar_1,
        "pred2similar_2": pred2similar_2
    }
    
    return results


def extract_attention_ensemble_word_semantics(input_ids, attention_matrix, tokenizer, shot):
    newline_id = 13
    newline_poss = [i for i, x in enumerate(input_ids.tolist()) if x == newline_id]

    word_pos = newline_poss[1] - 2
    similar_1_pos = newline_poss[1] - 3
    word_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[1 + i * 2] + 1, newline_poss[1 + i * 2 + 1])
        tgt_poss = range(newline_poss[1 + i * 2 + 1] + 1, newline_poss[1 + i * 2 + 2])
        word_example_poss  = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        word_examples_poss_list.append(word_example_poss)
    
    semantics_pos = newline_poss[2 + 2 * (shot // 2)] - 2
    similar_2_pos = newline_poss[2 + 2 * (shot // 2)] - 3
    semantics_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 1])
        tgt_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2 + 1] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 2])
        semantics_example_poss = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        semantics_examples_poss_list.append(semantics_example_poss)
    
    last_pos = len(input_ids) - 1

    worde2word = []
    worde2similar_1 = []
    semanticse2word = []
    semanticse2semantics = []
    semanticse2similar_1 = []
    semanticse2similar_2 = []
    pred2worde = []
    pred2semanticse = []
    pred2word = 0.0
    pred2semantics = 0.0
    pred2similar_1 = 0.0
    pred2similar_2 = 0.0

    for i in range(shot // 2):
        word_example_poss = word_examples_poss_list[i]
        attention_word_example_and_word = attention_matrix[word_example_poss, word_pos].mean().item()
        attention_prediction_and_word_example = attention_matrix[last_pos, word_example_poss].mean().item()
        attention_word_example_and_similar_1 = attention_matrix[word_example_poss, similar_1_pos].mean().item()
        worde2word.append(attention_word_example_and_word)
        pred2worde.append(attention_prediction_and_word_example)
        worde2similar_1.append(attention_word_example_and_similar_1)

        semantics_example_poss = semantics_examples_poss_list[i]
        attention_semantics_example_and_word = attention_matrix[semantics_example_poss, word_pos].mean().item()
        attention_semantics_example_and_semantics = attention_matrix[semantics_example_poss, semantics_pos].mean().item()
        attention_prediction_and_semantics_example = attention_matrix[last_pos, semantics_example_poss].mean().item()
        attention_semantics_example_and_similar_1 = attention_matrix[semantics_example_poss, similar_1_pos].mean().item()
        attention_semantics_example_and_similar_2 = attention_matrix[semantics_example_poss, similar_2_pos].mean().item()
        semanticse2word.append(attention_semantics_example_and_word)
        semanticse2semantics.append(attention_semantics_example_and_semantics)
        pred2semanticse.append(attention_prediction_and_semantics_example)
        semanticse2similar_1.append(attention_semantics_example_and_similar_1)
        semanticse2similar_2.append(attention_semantics_example_and_similar_2)
    
    pred2word = attention_matrix[last_pos, word_pos].item()
    pred2semantics = attention_matrix[last_pos, semantics_pos].item()
    pred2similar_1 = attention_matrix[last_pos, similar_1_pos].item()
    pred2similar_2 = attention_matrix[last_pos, similar_2_pos].item()

    results = {
        "worde2word": worde2word,
        "worde2similar_1": worde2similar_1,
        "semanticse2word": semanticse2word,
        "semanticse2semantics": semanticse2semantics,
        "semanticse2similar_1": semanticse2similar_1,
        "semanticse2similar_2": semanticse2similar_2,
        "pred2worde": pred2worde,
        "pred2semanticse": pred2semanticse,
        "pred2word": pred2word,
        "pred2semantics": pred2semantics,
        "pred2similar_1": pred2similar_1,
        "pred2similar_2": pred2similar_2
    }
    
    return results


def extract_attention_ensemble_random_random(input_ids, attention_matrix, tokenizer, shot):
    newline_id = 13
    newline_poss = [i for i, x in enumerate(input_ids.tolist()) if x == newline_id]

    r1_poss = range(newline_poss[0] + 5, newline_poss[1] - 1)
    similar_1_pos = newline_poss[0] + 4
    r1_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[1 + i * 2] + 1, newline_poss[1 + i * 2 + 1])
        tgt_poss = range(newline_poss[1 + i * 2 + 1] + 1, newline_poss[1 + i * 2 + 2])
        r1_example_poss  = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        r1_examples_poss_list.append(r1_example_poss)
    
    r2_poss = range(newline_poss[1 + 2 * (shot // 2)] + 5, newline_poss[2 + 2 * (shot // 2)] - 1)
    similar_2_pos = newline_poss[1 + 2 * (shot // 2)] + 4
    r2_examples_poss_list = []
    for i in range(shot // 2):
        src_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 1])
        tgt_poss = range(newline_poss[2 + 2 * (shot // 2) + i * 2 + 1] + 1, newline_poss[2 + 2 * (shot // 2) + i * 2 + 2])
        r2_example_poss = [pos for pos in src_poss] + [pos for pos in tgt_poss]
        r2_examples_poss_list.append(r2_example_poss)
    
    last_pos = len(input_ids) - 1

    r1e2r1 = []
    r1e2similar_1 = []
    r2e2r1 = []
    r2e2r2 = []
    r2e2similar_1 = []
    r2e2similar_2 = []
    pred2r1e = []
    pred2r2e = []
    pred2r1 = 0.0
    pred2r2 = 0.0
    pred2similar_1 = 0.0
    pred2similar_2 = 0.0

    for i in range(shot // 2):
        r1_example_poss = r1_examples_poss_list[i]
        attention_r1_example_and_r1 = 0.0
        for r1_pos in r1_poss:
            attention_r1_example_and_r1 += attention_matrix[r1_example_poss, r1_pos].mean().item()
        attention_r1_example_and_r1 /= len(r1_poss)
        attention_prediction_and_r1_example = attention_matrix[last_pos, r1_example_poss].mean().item()
        attention_r1_example_and_similar_1 = attention_matrix[r1_example_poss, similar_1_pos].mean().item()
        r1e2r1.append(attention_r1_example_and_r1)
        pred2r1e.append(attention_prediction_and_r1_example)
        r1e2similar_1.append(attention_r1_example_and_similar_1)

        r2_example_poss = r2_examples_poss_list[i]
        attention_r2_example_and_r1 = 0.0
        for r1_pos in r1_poss:
            attention_r2_example_and_r1 += attention_matrix[r2_example_poss, r1_pos].mean().item()
        attention_r2_example_and_r1 /= len(r1_poss)
        # attention_r2_example_and_r1 = attention_matrix[r2_example_poss, r1_poss].mean().item()
        attention_r2_example_and_r2 = 0.0
        for r2_pos in r2_poss:
            attention_r2_example_and_r2 += attention_matrix[r2_example_poss, r2_pos].mean().item()
        attention_r2_example_and_r2 /= len(r2_poss)
        # attention_r2_example_and_r2 = attention_matrix[r2_example_poss, r2_poss].mean().item()
        attention_prediction_and_r2_example = attention_matrix[last_pos, r2_example_poss].mean().item()
        attention_r2_example_and_similar_1 = attention_matrix[r2_example_poss, similar_1_pos].mean().item()
        attention_r2_example_and_similar_2 = attention_matrix[r2_example_poss, similar_2_pos].mean().item()
        r2e2r1.append(attention_r2_example_and_r1)
        r2e2r2.append(attention_r2_example_and_r2)
        pred2r2e.append(attention_prediction_and_r2_example)
        r2e2similar_1.append(attention_r2_example_and_similar_1)
        r2e2similar_2.append(attention_r2_example_and_similar_2)
    r1e2r1 = sum(r1e2r1) / len(r1e2r1)
    r2e2r1 = sum(r2e2r1) / len(r2e2r1)
    r2e2r2 = sum(r2e2r2) / len(r2e2r2)
    pred2r1 = attention_matrix[last_pos, r1_poss].mean().item()
    pred2r2 = attention_matrix[last_pos, r2_poss].mean().item()
    pred2similar_1 = attention_matrix[last_pos, similar_1_pos].item()
    pred2similar_2 = attention_matrix[last_pos, similar_2_pos].item()

    results = {
        "r1e2r1": r1e2r1,
        # "r1e2similar_1": r1e2similar_1,
        "r2e2r1": r2e2r1,
        "r2e2r2": r2e2r2,
        # "r2e2similar_1": r2e2similar_1,
        # "r2e2similar_2": r2e2similar_2,
        # "pred2r1e": pred2r1e,
        # "pred2r2e": pred2r2e,
        "pred2r1": pred2r1,
        "pred2r2": pred2r2,
        # "pred2similar_1": pred2similar_1,
        # "pred2similar_2": pred2similar_2
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


def main(selections=["bm25"], order="descending", langs=["de", "fr", "ru"], directions=["into", "outof"], shot=4, batch_size=4, templates=["a"], cut=-1, output_path="../result/alpaca/attention.tsv"):
    model_name_or_path = 'wxjiao/alpaca-7b'
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for template in templates:
        for selection in selections:
            results_sum_overalls = []

            for direction in directions:
                for lang in langs:
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
                            train_pairs = idx2example(train_sentence_pairs, idx_list[i][count_dict[sel] * (shot // len(idx_list_list)):], shot // len(idx_list_list), order)
                            train_pairs_list.append(train_pairs)
                            count_dict[sel] += 1
                        prompt = get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template=template, tokenizer=tokenizer)
                        prompts.append(prompt)
                    
                    results_sums = []

                    for i in tqdm(range(0, len(prompts), batch_size), ncols=60):
                        p = prompts[i:i+batch_size]
                        tokenized = tokenizer(p, padding=True, return_tensors="pt")
                        input_ids = tokenized.input_ids.cuda()
                        attn_mask = tokenized.attention_mask.cuda()
                        input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
                        attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

                        with torch.no_grad():
                            outputs = model(input_ids, attention_mask=attn_mask, output_attentions=True)
                            attentions = outputs.attentions

                            if results_sums == []:
                                results_sums = [{} for _ in range(len(attentions))]
                            
                            for layer in range(len(attentions)):
                                layer_attentions = attentions[layer]
                                results_sum = results_sums[layer]
                                for j, layer_attention in enumerate(layer_attentions):
                                    attention_matrix = layer_attention.mean(dim=0)
                                    curr_input_ids = input_ids[j]
                                    attention_matrix = attention_matrix.cpu()
                                    curr_input_ids = curr_input_ids.cpu()
                                    attention_results = extract_attention(curr_input_ids, attention_matrix, tokenizer, shot, template=template)
                                    for key in attention_results:
                                        if type(attention_results[key]) == list:
                                            if key not in results_sum:
                                                results_sum[key] = [0.0] * len(attention_results[key])
                                            for k in range(len(attention_results[key])):
                                                results_sum[key][k] += attention_results[key][k]
                                        else:
                                            if key not in results_sum:
                                                results_sum[key] = 0.0
                                            results_sum[key] += attention_results[key]

                    if len(results_sum_overalls) == 0:
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
            # print("-------------------------------------")
            # for key in results_sum_overall:
            #     if type(results_sum_overall[key]) == list:
            #         for i in range(len(results_sum_overall[key])):
            #             print(f"{key}_{i+1}: {10000*results_sum_overall[key][i]:.2f}")
            #     else:
            #         print(f"{key}: {10000*results_sum_overall[key]:.2f}")
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
    selections = ["bm25+polynomial"]
    order = "ascending"
    langs = ["de", "fr", "ru"]
    directions = ["into", "outof"]
    shot = 4
    batch_size = 8
    templates = ["ensemble_word_syntax", "ensemble_random_random"]
    cut = -1
    main(selections, order, langs, directions, shot, batch_size, templates, cut)