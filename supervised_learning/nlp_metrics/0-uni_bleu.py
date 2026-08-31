#!/usr/bin/env python3
"""Unigram BLEU score"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence

    Args:
        references: list of reference translations
            each reference translation is a list of words
            in the translation
        sentence: list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    sentence_len = len(sentence)
    sentence_counts = {}
    for word in sentence:
        sentence_counts[word] = sentence_counts.get(word, 0) + 1

    max_ref_counts = {}
    for ref in references:
        ref_counts = {}
        for word in ref:
            ref_counts[word] = ref_counts.get(word, 0) + 1
        for word, count in ref_counts.items():
            max_ref_counts[word] = max(max_ref_counts.get(word, 0), count)

    clipped_count = 0
    for word, count in sentence_counts.items():
        clipped_count += min(count, max_ref_counts.get(word, 0))

    precision = clipped_count / sentence_len

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - sentence_len), ref_len))

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - (closest_ref_len / sentence_len))

    return bp * precision
