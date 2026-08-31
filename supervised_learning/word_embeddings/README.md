# Natural Language Processing - Word Embeddings

This project implements word embedding techniques used in Natural
Language Processing (NLP), starting with a Bag Of Words model.

## Tasks

### 0. Bag Of Words

File: `0-bag_of_words.py`

Contains the function `def bag_of_words(sentences, vocab=None):` that
builds a Bag Of Words embedding matrix from a list of sentences.

**Parameters:**
- `sentences` — a list of sentences to analyze
- `vocab` — a list of vocabulary words to use for the analysis. If
  `None`, all words found within `sentences` are used.

**Returns:** `embeddings, features`
- `embeddings` — a `numpy.ndarray` of shape `(s, f)` containing the
  word-count embeddings, where `s` is the number of sentences and `f`
  is the number of features (vocabulary words)
- `features` — a list of the feature (vocabulary) words used, in
  alphabetical order

**Implementation notes:**
- Uses `sklearn.feature_extraction.text.CountVectorizer` to build the
  word-count matrix (the `gensim` library is not used, per the task
  requirements).
- Text is automatically lowercased, punctuation is stripped, and
  single-character tokens are ignored by `CountVectorizer`'s default
  tokenizer, which matches the expected example output (e.g.
  `children's` is counted simply as `children`).

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- numpy
- scikit-learn

## Installation

pip install numpy scikit-learn --break-system-packages

## Usage

chmod +x 0-bag_of_words.py
./0-main.py

## Author

Holberton School - Machine Learning Specialization
