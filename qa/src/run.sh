python qa_inference.py \
    --device 0 \
    --dataset csqa strategyqa date sports logicalfallacy threeobjects knownunknowns gsm8k aqua \
    --shot 4 \
    --batch_size 4 \
    --templates ensemble_random \
    --cot_mode 2 \
    --max_new_tokens 256 \
    --models alpaca llama2 mistral

python qa_eval.py