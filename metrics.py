import numpy as np
import tensorflow as tf
import tensorflow_text
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu


def masked_accuracy(y_true, pred_logits):
    accuracies = tf.equal(y_true, tf.cast(tf.argmax(pred_logits, axis=2), tf.float32))

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def tensors_to_text(y_true: tf.Tensor, pred_logits: tf.Tensor, target_tokenizer: tensorflow_text.Tokenizer):

    y_pred = tf.argmax(pred_logits, axis=2)
    y_true = tf.cast(y_true, tf.int64)
    y_pred = target_tokenizer.detokenize(tf.RaggedTensor.from_tensor(y_pred, padding=0)).numpy()  # (b, ) string
    y_true = target_tokenizer.detokenize(tf.RaggedTensor.from_tensor(y_true, padding=0)).numpy()

    y_pred = [sentence.decode('utf-8').split() for sentence in y_pred]  # List[List[str]]
    y_true = [sentence.decode('utf-8').split() for sentence in y_true]
    return y_true, y_pred


def bleu_score(target_tokenizer: tensorflow_text.Tokenizer, order: int):
    def _bleu_score(y_true: tf.Tensor, pred_logits: tf.Tensor):

        y_true, y_pred = tensors_to_text(y_true, pred_logits, target_tokenizer)
        bleus = []

        for yt, yp in zip(y_true, y_pred):
            bleus.append(sentence_bleu([yt], yp[:len(yt)], weights=np.ones(order) / order))
        return np.mean(bleus)

    _bleu_score.__name__ = f'bleu_{order}'

    return _bleu_score


def gleu_score(target_tokenizer: tensorflow_text.Tokenizer):
    def _gleu_score(y_true: tf.Tensor, pred_logits: tf.Tensor):

        y_true, y_pred = tensors_to_text(y_true, pred_logits, target_tokenizer)
        bleus = []

        for yt, yp in zip(y_true, y_pred):
            bleus.append(sentence_gleu([yt], yp[:len(yt)]))
        return np.mean(bleus)

    _gleu_score.__name__ = f'gleu'

    return _gleu_score
