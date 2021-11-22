import tensorflow as tf
import numpy as np

from typing import *


def scaled_dot_product_attention(
        q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: Optional[tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to matmul_qk below, hence (..., seq_len_q, seq_len_k).

    Returns:
    output, attention_weights
    """

    # assert tf.shape(q)[-1] == tf.shape(k)[-1]
    # assert tf.shape(k)[-2] == tf.shape(v)[-2]

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor. Where mask=1, a large negative number is added to logits making the
    # activations which follow practically 0.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (b, seq_len, d_model)
        k = self.wk(k)  # (b, seq_len, d_model)
        v = self.wv(v)  # (b, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (b, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (b, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (b, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (b, num_heads, seq_len_q, depth)
        # attention_weights.shape == (b, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (b, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
# y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
# out, attn = temp_mha(y, k=y, q=y, mask=None)
# print(out.shape, attn.shape)


def point_wise_feed_forward_network(d_model: int, dff: int):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: Optional[tf.Tensor]) -> tf.Tensor:

        attn_output, _ = self.multi_head_attention(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.feed_forward(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff, dropout_rate: float = 0.1):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention_1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention_2 = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(
            self,
            x: tf.Tensor, enc_output: tf.Tensor, training: bool, look_ahead_mask: tf.Tensor, padding_mask: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # This is the output sequence's self-attention. look_ahead_mask prevents from using "future" tokens.
        # attn1 shape is (b, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.multi_head_attention_1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # This next one is when key, value are both from the encoder, and query is from the decoder.
        # This part is where tokens of the generate sequence decide to pay attention to the input sequence.
        # padding_mask prevents the output sequence from attending to a lot of padded values in the input sequence.
        attn2, attn_weights_block2 = self.multi_head_attention_2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.feed_forward(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# sample_encoder_layer = EncoderLayer(512, 8, 2048)
#
# sample_encoder_layer_output = sample_encoder_layer(
#     tf.random.uniform((64, 43, 512)), False, None)
#
# print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
#
#
# sample_decoder_layer = DecoderLayer(512, 8, 2048)
#
# sample_decoder_layer_output, _, _ = sample_decoder_layer(
#     tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
#     False, None, None)
#
# print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)


def positional_encoding(position: int, d_model: int):

    def get_angles(pos: np.ndarray, i: np.ndarray) -> np.ndarray:
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],  # (position, 1)
        np.arange(d_model)[np.newaxis, :],  # (1, d_model)
    )  # (position, d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)  # (1, position, d_model)

# n, d = 2048, 512
# pos_encoding = positional_encoding(n, d)
# print(pos_encoding.shape)
# pos_encoding = pos_encoding[0]
#
# # Juggle the dimensions for the plot
# pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
# pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
# pos_encoding = tf.reshape(pos_encoding, (d, n))
#
# plt.pcolormesh(pos_encoding, cmap='RdBu')
# plt.ylabel('Depth')
# plt.xlabel('Position')
# plt.colorbar()
# plt.show()


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 input_vocab_size: int,
                 maximum_position_encoding: int,
                 dropout_rate: float = 0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # positional encoding has shape (1, maximum_position_encoding, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

        # why is this happening? (probably explained in the paper)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
#                          dff=2048, input_vocab_size=8500,
#                          maximum_position_encoding=10000)
# temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
#
# sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
#
# print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 target_vocab_size: int,
                 maximum_position_encoding: int,
                 dropout_rate: float = 0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self,
             x: tf.Tensor,
             enc_output: tf.Tensor,
             training: bool,
             look_ahead_mask: tf.Tensor,
             padding_mask: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.decoder_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
#                          dff=2048, target_vocab_size=8000,
#                          maximum_position_encoding=5000)
# temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
#
# output, attn = sample_decoder(temp_input,
#                               enc_output=sample_encoder_output,
#                               training=False,
#                               look_ahead_mask=None,
#                               padding_mask=None)
#
# print(output.shape, attn['decoder_layer2_block2'].shape)


def create_padding_mask(seq: tf.Tensor) -> tf.Tensor:
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    TODO: I don't think this is correct. It should return the complement of the upper triangle.
    returns the complement of the lower-triangle of shape (size, size)
    [[0, 1, 1, ..., 1],
     [0, 0, 1, ..., 1],
     ...
     [0, 0, 0, ..., 0]]
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class TransformerFirst(tf.keras.Model):
    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 input_vocab_size: int,
                 target_vocab_size: int,
                 max_posit_encode_input: int,
                 max_posit_encode_target: int,
                 dropout_rate: float = 0.1
                 ):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_posit_encode_input, dropout_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_posit_encode_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        # TODO: Unclear why this has to be separate from the one above
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


def create_transformer(
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        max_posit_encode_input: int,
        max_posit_encode_target: int,
        dropout_rate: float = 0.1
) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(None, ))
    tar = tf.keras.layers.Input(shape=(None, ))

    encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_posit_encode_input, dropout_rate)

    decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_posit_encode_target, dropout_rate)

    final_layer = tf.keras.layers.Dense(target_vocab_size)

    enc_padding_mask, look_ahead_mask, dec_padding_mask = TransformerFirst.create_masks(inp, tar)

    enc_output = encoder(inp, True, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = decoder(tar, enc_output, True, look_ahead_mask, dec_padding_mask)

    final_output = final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return tf.keras.Model(inputs={"input": inp, "target": tar}, outputs=final_output)


class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 input_vocab_size: int,
                 target_vocab_size: int,
                 max_posit_encode_input: int,
                 max_posit_encode_target: int,
                 dropout_rate: float = 0.1
                 ):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_posit_encode_input, dropout_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_posit_encode_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool) -> tf.Tensor:
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs['input'], inputs['target']

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        # TODO: Unclear why this has to be separate from the one above
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


