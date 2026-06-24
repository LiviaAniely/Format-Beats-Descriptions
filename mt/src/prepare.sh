# dependency parsing for Polynomial
if [ -f ../data/de/train.de.spacy.json ] && [ -f ../data/fr/train.fr.spacy.json ] && [ -f ../data/ru/train.ru.spacy.json ]; then
    echo "SpaCy parse files already exist. Skipping parse.py."
else
    echo "Running parse.py..."
    python parse.py
fi

# generate random examples
if [ -f ../data/de/index/test/into/rand1.index ] && [ -f ../data/fr/index/test/into/rand1.index ] && [ -f ../data/ru/index/test/into/rand1.index ]; then
    echo "Random examples index files already exist. Skipping rand_example.py."
else
    echo "Running rand_example.py..."
    python rand_example.py
fi

# get BM25 examples
if [ -f ../data/de/index/test/into/bm25.index ] && [ -f ../data/fr/index/test/into/bm25.index ] && [ -f ../data/ru/index/test/into/bm25.index ]; then
    echo "BM25 index files already exist. Skipping bm25.py."
else
    echo "Running bm25.py..."
    python bm25.py
fi

# get Polynomial examples
if [ -f ../data/de/index/test/into/polynomial.index ] && [ -f ../data/fr/index/test/into/polynomial.index ] && [ -f ../data/ru/index/test/into/polynomial.index ]; then
    echo "Polynomial index files already exist. Skipping polynomial.py."
else
    echo "Running polynomial.py..."
    python polynomial.py
fi

# generate random nouns
if [ -f noun_65536.txt ]; then
    echo "Random nouns file already exists. Skipping rand_noun.py."
else
    echo "Running rand_noun.py..."
    python rand_noun.py
fi