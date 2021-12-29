import tensorflow as tf


class Translator(tf.Module):
    def __init__(self, input_tokenizer, target_tokenizer, transformer):
        super(Translator, self).__init__(name='translator')
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.transformer = transformer

    def __call__(self, sentence, max_length: int):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        # adding the start- and end-of-sequence tokens to the input
        encoder_input = self.input_tokenizer.tokenize(sentence).to_tensor()

        # get start- and end-of-sequence tokens for the target language
        start_end = self.target_tokenizer.tokenize([''])[0]
        start_token = start_end[0][tf.newaxis]
        end_token = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start_token)

        # generate one output word at a time until we reach the end-of-sentence token.
        # at each step the single most probable predicted token is picked.
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer({'input': encoder_input, 'target': output}, training=False)

            # select the last token from the seq_len dimension, others are kept from previous steps
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end_token:
                break

        output = tf.transpose(output_array.stack())  # output.shape (1, tokens)
        text = self.target_tokenizer.detokenize(output)[0]  # shape: ()

        tokens = self.target_tokenizer.lookup(output)[0]

        return text, tokens
