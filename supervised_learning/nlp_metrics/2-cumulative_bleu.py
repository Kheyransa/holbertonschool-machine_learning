#!/usr/bin/env python3
"""Cumulative N-gram BLEU score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence

    Args:
        references: list of reference translations
            each reference translation is a list of words
            in the translation
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        the cumulative n-gram BLEU score
    """
    def get_ngrams(words, k):
        return [tuple(words[i:i + k]) for i in range(len(words) - k + 1)]

    def ngram_precision(k):
        sentence_ngrams = get_ngrams(sentence, k)
        sentence_len = len(sentence_ngrams)

        sentence_counts = {}
        for gram in sentence_ngrams:
            sentence_counts[gram] = sentence_counts.get(gram, 0) + 1

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = get_ngrams(ref, k)
            ref_counts = {}
            for gram in ref_ngrams:
                ref_counts[gram] = ref_counts.get(gram, 0) + 1
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(
                    max_ref_counts.get(gram, 0), count)

        clipped_count = 0
        for gram, count in sentence_counts.items():
            clipped_count += min(count, max_ref_counts.get(gram, 0))

        return clipped_count / sentence_len

    precisions = [ngram_precision(k) for k in range(1, n + 1)]
    geo_mean = np.exp(np.sum(np.log(precisions)) / n)

    sentence_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - sentence_len), ref_len))

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - (closest_ref_len / sentence_len))

    return bp * geo_mean
