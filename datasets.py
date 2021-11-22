"""
pip install tensorflow_datasets
pip install -U tensorflow-text

"""
import logging
import os.path

import tensorflow_datasets as tfds
import tensorflow_text
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

BUFFER_SIZE = 20000


def cache_model(
        model_name,
        cache_dir,
        cache_subdir="",
):

    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir=cache_dir,
        cache_subdir=cache_subdir,
        extract=True
    )


def translation_inputs_targets(
        ds: tf.data.Dataset,
        tokenizer_input: tensorflow_text.Tokenizer,
        tokenizer_target: tensorflow_text.Tokenizer,
        batch_size: int,
):
    def _tokenize_pairs(inp: tf.Tensor, tar: tf.Tensor):

        inp_token = tokenizer_input.tokenize(inp)
        # Convert from ragged to dense, padding with zeros.
        inp_token = inp_token.to_tensor()

        tar_token = tokenizer_target.tokenize(tar)
        tar_token = tar_token.to_tensor()

        return {'input': inp_token, 'target': tar_token[:, :-1]}, tar_token[:, 1:]

    return ds.cache().batch(batch_size).map(
        _tokenize_pairs,
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).repeat().prefetch(tf.data.AUTOTUNE)


def tokenizers_from_tf_model(model_name: str, cache_dir: str):

    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir=cache_dir,
        cache_subdir="",
        extract=True
    )

    tokenizers = tf.saved_model.load(os.path.join(cache_dir, model_name))
    return tokenizers


def pt_to_en_tokenizers(cache_dir):
    model_name = "ted_hrlr_translate_pt_en_converter"
    tokenizers = tokenizers_from_tf_model(model_name, cache_dir)
    return tokenizers.pt, tokenizers.en


def translation_train_and_val_from_tfds(
        tf_dataset_name: str,
        batch_size: int,
        tokenizer_input: tensorflow_text.Tokenizer,
        tokenizer_target: tensorflow_text.Tokenizer,
):
    examples, metadata = tfds.load(tf_dataset_name, with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    train_batches = translation_inputs_targets(train_examples, tokenizer_input, tokenizer_target, batch_size)
    val_batches = translation_inputs_targets(val_examples, tokenizer_input, tokenizer_target, batch_size)
    return train_batches, val_batches
