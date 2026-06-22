# Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions

[![arXiv](https://img.shields.io/badge/EMNLP_2025_Finding-3-red.svg)](https://aclanthology.org/2025.findings-emnlp.3/)
[![arXiv](https://img.shields.io/badge/arXiv-2408.08780-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2408.08780)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?logo=apache)](LICENSE)

Source code of our EMNLP 2025 Findings paper **Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions**.

## Prerequisites
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download ru_core_news_sm
```

<!-- Comandos para baixar os datasets -->
<!-- pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl -->

<!-- pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl -->

<!-- pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl -->

<!-- pip install https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.7.0/ru_core_news_sm-3.7.0-py3-none-any.whl -->

<!-- Teste para verificar se os datasets baixaram -->
<!-- python -c "import spacy; [spacy.load(m) for m in ['en_core_web_sm', 'de_core_news_sm', 'fr_core_news_sm', 'ru_core_news_sm']]; print('Todos os 4 modelos foram carregados com sucesso')"
 -->
## Machine Translation (MT)

```bash
cd mt
```

### Preparation
Fetch and extract the training data. Note that test data have been provided for convenience.

```bash
cd data
sh prepare.sh
```

Prepare for in-context examples. Note that `retriv` might be incompatible with `torch` of some versions in some cases. We recommend to run `bm25.py` in another environment if this occurs.

```bash
cd ../src
sh prepare.sh
```

### Run Experiments and Evaluation
```bash
sh run.sh
```

## QA Tasks

### Data

The datasets corresponding to each task appearing in our paper have been placed in [data](). The file names of the training set (example database) and test set must contain *train* and *test* or *dev* respectively, and both end with *json* or *jsonl*.

### Example of Runing Experiments

```bash
cd qa/src

python qa_inference.py \
    --device 0 \
    --dataset csqa strategyqa date sports logicalfallacy threeobjects knownunknowns gsm8k aqua \
    --shot 4 \
    --batch_size 32 \
    --templates vanilla ensemble_random \
    --cot_mode 1 \
    --max_new_tokens 256 \
    --models alpaca llama2 mistral
```

For OpenAI's API model, you can refer to [qa_gpt_inference.py]().

### Evaluation

In order to evaluate our proposed prompt template in different dimensions, we set up two forms of evaluation, one is cross-model and the other is cross-dataset. Please refer to [qa_eval.py]() for details.

### How to Add New Datasets

1. Refer to the architecture (optional) and format of the existing datasets and put your new dataset into [data]().
2. Inherit `ReasoningData` defined in [data_loader.py]() and overwrite corresponding methods according to your new dataset.
3. Add `@dateset_register` for your child class.

## Citation
If you find our work helpful, feel free to cite us:
```
@inproceedings{tang-etal-2025-large-language,
    title = "Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions",
    author = "Tang, Chenming  and
      Wang, Zhixiang  and
      Sun, Hao  and
      Wu, Yunfang",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.3/",
    doi = "10.18653/v1/2025.findings-emnlp.3",
    pages = "26--48",
    ISBN = "979-8-89176-335-7",
    abstract = "With the help of in-context learning (ICL), large language models (LLMs) have achieved impressive performance across various tasks. However, the function of descriptive instructions during ICL remains under-explored. In this work, we propose an ensemble prompt framework to describe the selection criteria of multiple in-context examples, and preliminary experiments on machine translation (MT) across six translation directions confirm that this framework boosts ICL performance. But to our surprise, LLMs might not care what the descriptions actually say, and the performance gain is primarily caused by the ensemble format, since it could lead to improvement even with random descriptive nouns. We further apply this new ensemble framework on a range of commonsense, math, logical reasoning and hallucination tasks with three LLMs and achieve promising results, suggesting again that designing a proper prompt format would be much more effective and efficient than paying effort into specific descriptions."
}
```
