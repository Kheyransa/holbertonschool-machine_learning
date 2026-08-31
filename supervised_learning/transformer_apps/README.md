# Transformer Applications

This project implements machine translation with a transformer model,
using a Portuguese-to-English dataset.

## Setup

TFDS can no longer download `ted_hrlr_translate/pt_to_en` because the
upstream archive is offline. Before running any file in this project:

1. Download and extract the dataset:
```bash
curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/ted_hrlr_pt_to_en.tar.gz
mkdir -p ~/.cache/ted_hrlr/
tar -xzf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr/
```

2. Download `setup.py` to the root of the project folder:
```bash
curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/setup.py
```

3. Use `load_pt2en(split)` (imported from `setup`) wherever
   `tfds.load('ted_hrlr_translate/pt_to_en', split=split, as_supervised=True)`
   would have been used. The return type is identical: a
   `tf.data.Dataset` of `(pt, en)` `tf.string` pairs.

## Files

### 0-dataset.py
Contains the class `Dataset` that loads and preps a dataset for
machine translation.

**Class constructor** `def __init__(self):`
Creates the instance attributes:
- `data_train` - the `ted_hrlr_translate/pt_to_en` `train` split as a
  `tf.data.Dataset`, loaded via `load_pt2en('train')`
- `data_valid` - the `ted_hrlr_translate/pt_to_en` `validation` split
  as a `tf.data.Dataset`, loaded via `load_pt2en('validation')`
- `tokenizer_pt` - the Portuguese tokenizer created from the training
  set
- `tokenizer_en` - the English tokenizer created from the training
  set

**Instance method** `def tokenize_dataset(self, data):`
Creates sub-word tokenizers for the dataset.
- `data` is a `tf.data.Dataset` whose examples are formatted as a
  tuple `(pt, en)`
  - `pt` is the `tf.Tensor` containing the Portuguese sentence
  - `en` is the `tf.Tensor` containing the corresponding English
    sentence
- Uses a pre-trained tokenizer:
  - `neuralmind/bert-base-portuguese-cased` for the Portuguese text
  - `bert-base-uncased` for the English text
- Trains the tokenizers with a maximum vocabulary size of `2**13`
- Returns: `tokenizer_pt, tokenizer_en`

## Requirements
- Python 3.6+
- TensorFlow
- transformers

## Usage
```python
Dataset = __import__('0-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
```

### 1-dataset.py
Updates the class `Dataset` from `0-dataset.py`.

**Instance method** `def encode(self, pt, en):`
Encodes a translation into tokens.
- `pt` is the `tf.Tensor` containing the Portuguese sentence
- `en` is the `tf.Tensor` containing the corresponding English
  sentence
- The tokenized sentences include the start and end of sentence
  tokens
  - The start token is indexed as `vocab_size`
  - The end token is indexed as `vocab_size + 1`
- Returns: `pt_tokens, en_tokens`
  - `pt_tokens` is a `list` containing the Portuguese tokens
  - `en_tokens` is a `list` containing the English tokens

### 2-dataset.py
Updates the class `Dataset` from `1-dataset.py`.

**Instance method** `def tf_encode(self, pt, en):`
Acts as a `tensorflow` wrapper for the `encode` instance method. Sets
the shape of the `pt` and `en` return tensors.

**Class constructor** `def __init__(self):`
Updated to tokenize the examples in `data_train` and `data_valid`
using `tf_encode`.

### 3-dataset.py
Updates the class `Dataset` from `2-dataset.py` to set up the data
pipeline.

**Class constructor** `def __init__(self, batch_size, max_len):`
- `batch_size` is the batch size for training/validation
- `max_len` is the maximum number of tokens allowed per example
  sentence

Updates the `data_train` attribute by:
- filtering out all examples that have either sentence with more than
  `max_len` tokens
- caching the dataset to increase performance
- shuffling the entire dataset using a buffer size equal to `20000`
- splitting the dataset into padded batches of size `batch_size`
- prefetching the dataset using `tf.data.experimental.AUTOTUNE` to
  increase performance

Updates the `data_valid` attribute by:
- filtering out all examples that have either sentence with more than
  `max_len` tokens
- splitting the dataset into padded batches of size `batch_size`

### 4-create_masks.py
Contains the function `create_masks(inputs, target)` that creates all
masks for training/validation.

- `inputs` is a tf.Tensor of shape `(batch_size, seq_len_in)` that
  contains the input sentence
- `target` is a tf.Tensor of shape `(batch_size, seq_len_out)` that
  contains the target sentence
- Only uses `tensorflow` operations in order to properly function in
  the training step
- Returns: `encoder_mask, combined_mask, decoder_mask`
  - `encoder_mask` is the tf.Tensor padding mask of shape
    `(batch_size, 1, 1, seq_len_in)` to be applied in the encoder
  - `combined_mask` is the tf.Tensor of shape
    `(batch_size, 1, seq_len_out, seq_len_out)` used in the 1st
    attention block in the decoder to pad and mask future tokens in
    the input received by the decoder. It takes the maximum between
    a look ahead mask and the decoder target padding mask.
  - `decoder_mask` is the tf.Tensor padding mask of shape
    `(batch_size, 1, 1, seq_len_in)` used in the 2nd attention block
    in the decoder

### 4-create_masks.py
Contains the function `create_masks(inputs, target)` that creates all
masks for training/validation.

- `inputs` is a tf.Tensor of shape `(batch_size, seq_len_in)` that
  contains the input sentence
- `target` is a tf.Tensor of shape `(batch_size, seq_len_out)` that
  contains the target sentence
- Only uses `tensorflow` operations in order to properly function in
  the training step
- Returns: `encoder_mask, combined_mask, decoder_mask`
  - `encoder_mask` is the tf.Tensor padding mask of shape
    `(batch_size, 1, 1, seq_len_in)` to be applied in the encoder
  - `combined_mask` is the tf.Tensor of shape
    `(batch_size, 1, seq_len_out, seq_len_out)` used in the 1st
    attention block in the decoder to pad and mask future tokens in
    the input received by the decoder. It takes the maximum between
    a look ahead mask and the decoder target padding mask.
  - `decoder_mask` is the tf.Tensor padding mask of shape
    `(batch_size, 1, 1, seq_len_in)` used in the 2nd attention block
    in the decoder
