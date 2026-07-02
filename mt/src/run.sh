export CUDA_VISIBLE_DEVICES=0

rm -f ../output/current_run_suffix.txt

# Inference (XGLM)
python xglm.py

# Inference (Alpaca)
python alpaca.py

# Evaluation (COMET-20)
python eval.py

# Attention (XGLM)
python attention_xglm.py

# Attention (Alpaca)
python attention_alpaca.py