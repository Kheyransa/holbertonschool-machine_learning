# NLP Metrics

This project implements NLP evaluation metrics, starting with BLEU score.

## Files

### 0-uni_bleu.py
Contains the function `uni_bleu(references, sentence)` that calculates
the unigram BLEU score for a sentence.

- `references` is a list of reference translations, each a list of words
- `sentence` is a list containing the model proposed sentence
- Returns the unigram BLEU score

### 1-ngram_bleu.py
Contains the function `ngram_bleu(references, sentence, n)` that
calculates the n-gram BLEU score for a sentence.

- `references` is a list of reference translations, each a list of words
- `sentence` is a list containing the model proposed sentence
- `n` is the size of the n-gram to use for evaluation
- Returns the n-gram BLEU score

### 2-cumulative_bleu.py
Contains the function `cumulative_bleu(references, sentence, n)` that
calculates the cumulative n-gram BLEU score for a sentence.

- `references` is a list of reference translations, each a list of words
- `sentence` is a list containing the model proposed sentence
- `n` is the size of the largest n-gram to use for evaluation
- All n-gram scores are weighted evenly
- Returns the cumulative n-gram BLEU score

## Requirements
- Python 3.6+
- numpy

## Usage
```python
uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"],
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
# 0.6549846024623855
```
