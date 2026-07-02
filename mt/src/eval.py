import torch
import os

from score import init_comet_20, get_comet_20_mean_score, get_bleu_mean_score

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

def main(model="alpaca", device="cuda", selections=["bm25-polynomial"], order="descending", langs=["de", "fr", "ru"], directions=["into", "outof"], shot=4, templates=["alpaca"], log_path="../result/alpaca/result.tsv"):
    torch.device(device)
    comet_20_metric = init_comet_20("Unbabel/wmt20-comet-da")

    for template in templates:
        for direction in directions:
            for lang in langs:
                for selection in selections:
                    if direction == "into":
                        test_src_fn = f"../data/{lang}/test.{lang}"
                        test_tgt_fn = f"../data/{lang}/test.en"
                    else:
                        test_src_fn = f"../data/{lang}/test.en"
                        test_tgt_fn = f"../data/{lang}/test.{lang}"

                    test_sentences = []
                    with open(test_src_fn, "r") as f:
                        for line in f:
                            test_sentences.append(line.strip())
                    gold = []
                    with open(test_tgt_fn, "r") as f:
                        for line in f:
                            gold.append(line.strip())
                    
                    suffix = get_run_suffix(is_inference=False)
                    output_fn = f"../output/{model}{suffix}/{lang}.{direction}.{selection}.{shot}.{order}.{template}.txt"

                    system = []
                    with open(output_fn, "r") as f:
                        for line in f:
                            system.append(line.strip())
                    
                    source = test_sentences

                    comet_20_score = get_comet_20_mean_score(system, gold, source, comet_20_metric)
                    bleu_score = get_bleu_mean_score(system, gold)
                    print("=====================================")
                    print(f"Language: {lang}")
                    print(f"Direction: {direction}")
                    print(f"Selection: {selection}")
                    print(f"Shot: {shot}")
                    print(f"Order: {order}")
                    print(f"Template: {template}")
                    print('-------------------------------------')
                    print(f"COMET-20: {100 * comet_20_score}")
                    print(f"BLEU: {bleu_score}")
                    print("=====================================")

                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, 'a') as f_log:
                        f_log.write(f"{template}\t{shot}\t{lang}\t{direction}\t{selection}\t{order}\t{100 * comet_20_score}\t{bleu_score}\n")
                    f_log.close()


if __name__ == "__main__":
    device = "cuda"
    selections = ["rand1+rand1", "bm25+polynomial", "polynomial+bm25"]
    order = "ascending"
    langs = ["de", "fr", "ru"]
    directions = ["into"]
    shot = 4
    suffix = get_run_suffix(is_inference=False)
    model = "xglm"
    templates = ["vanilla", "ensemble_word_syntax", "ensemble_random_random"]
    log_path = f"../result/{model}{suffix}/result.tsv"
    main(model, device, selections, order, langs, directions, shot, templates, log_path)

    model = "alpaca"
    templates = ["alpaca_ensemble", "ensemble_word_syntax", "ensemble_random_random"]
    log_path = f"../result/{model}{suffix}/result.tsv"
    main(model, device, selections, order, langs, directions, shot, templates, log_path)