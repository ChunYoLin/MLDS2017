import tensorflow as tf
import numpy as np

class S2VT_model(object):
    def __init__(self):
        size = 200
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(size, forget_bias = 0., state_is_tuple = True)
        #  encoder top
        encoder_cell_top = lstm_cell()
        input_frame = tf.placeholder(dtype = tf.float32, shape = [80, 4096])
        top_state_in = encoder_cell_top.zero_state(1, tf.float32)
        top_outputs = []
        with tf.variable_scope("encoder_top"):
            for time_step in range(80):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                top_input = tf.reshape(input_frame[time_step], [1, 4096])
                (top_output, top_state_out) = encoder_cell_top(top_input, top_state_in)
                pad = tf.zeros(shape = [1, size])
                top_output_pad = tf.concat(values = [pad, top_output], axis = 1)
                top_state_in = top_state_out
                top_outputs.append(top_output_pad)
        self._top_final_state = top_state_out

        #  encoder bot
        encoder_cell_bot = lstm_cell()
        bot_state_in = encoder_cell_bot.zero_state(1, tf.float32)
        with tf.variable_scope("encoder_bot"):
            for time_step in range(80):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                bot_input = top_outputs[time_step]
                (bot_output, bot_state_out) = encoder_cell_bot(bot_input, bot_state_in)
                bot_state_in = bot_state_out
        self._bot_final_state = bot_state_out

        #  decoder top
        decoder_cell_top = lstm_cell()
        pad = tf.zeros(shape = [caption_length, 4096])
        decoder_output_top, decoder_top_state = tf.nn.dynamic_rnn(
                cell = decoder_cell_top, 
                inputs = pad, 
                initial_state = self._top_final_state)

a = S2VT_model()
