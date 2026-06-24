# From CTQScorer (Kumar et al., 2023)
# https://github.com/AI4Bharat/CTQScorer/blob/master/dataset/prepare_datasets.sh

# Check if the final extracted training files already exist
if [ -f de/train.de ] && [ -f fr/train.fr ] && [ -f ru/train.ru ]; then
    echo "Training datasets are already extracted and prepared. Skipping download and extraction."
    exit 0
fi

# Download datasets only if neither the decompressed tsv/txt nor the compressed .gz files exist
if [ ! -f europarl-v10.fr-en.tsv ] && [ ! -f europarl-v10.fr-en.tsv.gz ]; then
    echo "Downloading europarl-v10.fr-en.tsv.gz..."
    wget https://www.statmt.org/europarl/v10/training/europarl-v10.fr-en.tsv.gz
fi

if [ ! -f europarl-v10.de-en.tsv ] && [ ! -f europarl-v10.de-en.tsv.gz ]; then
    echo "Downloading europarl-v10.de-en.tsv.gz..."
    wget https://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz
fi

if [ ! -f en-ru.txt ] && [ ! -f en-ru.txt.gz ]; then
    echo "Downloading en-ru.txt.gz..."
    wget https://s3.amazonaws.com/web-language-models/paracrawl/bonus/en-ru.txt.gz
fi

# Decompress files if the .gz files exist and we need the raw files
if [ -f europarl-v10.fr-en.tsv.gz ]; then
    echo "Decompressing europarl-v10.fr-en.tsv.gz..."
    gzip -d europarl-v10.fr-en.tsv.gz
fi

if [ -f europarl-v10.de-en.tsv.gz ]; then
    echo "Decompressing europarl-v10.de-en.tsv.gz..."
    gzip -d europarl-v10.de-en.tsv.gz
fi

if [ -f en-ru.txt.gz ]; then
    echo "Decompressing en-ru.txt.gz..."
    gzip -d en-ru.txt.gz
fi

# Run python extraction script
if [ -f europarl-v10.fr-en.tsv ] && [ -f europarl-v10.de-en.tsv ] && [ -f en-ru.txt ]; then
    echo "Extracting datasets..."
    python extract.py
else
    echo "Error: TSV/TXT files for extraction not found!"
    exit 1
fi