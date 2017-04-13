import tensorflow as tf
import numpy as np
import data_reader

class S2VT_input(object):
  def __init__(self, frame_data, text_data, sent_len, word_id, name = None):
    batch_size = 1
    self.frame_len = 80
    self.sent_len = sent_len
    self.vocab_size = len(word_id)
    self.batch_size = batch_size
    self.num_steps = self.frame_len + self.sent_len
    self.epoch_size = len(frame_data) // batch_size
    self.input_batch, self.targets_batch = data_reader.Data_producer(
            frame_data = frame_data, 
            text_data = text_data, 
            batch_size = batch_size, 
            sent_len = sent_len)

class S2VT_model(object):
    def __init__(self, is_training, input_):
        size = 256
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(size, forget_bias = 0., state_is_tuple = True)
        frame_len = input_.frame_len
        sent_len = input_.sent_len
        vocab_size = input_.vocab_size
        batch_size = input_.batch_size
        num_steps = frame_len + sent_len
        # top
        cell_top = lstm_cell()
        input_frame = input_.input_batch
        top_state_in = cell_top.zero_state(batch_size, tf.float32)
        top_outputs = []
        with tf.variable_scope("top_cell"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                if time_step < frame_len:
                    top_input = input_frame[:, time_step, :]
                else:
                    top_input = tf.zeros(shape = [batch_size, 4096])
                (top_output, top_state_out) = cell_top(top_input, top_state_in)
                pad = tf.zeros(shape = [batch_size, size])
                top_output_pad = tf.concat(values = [pad, top_output], axis = 1)
                top_state_in = top_state_out
                top_outputs.append(top_output_pad)
        #  bot
        cell_bot = lstm_cell()
        bot_state_in = cell_bot.zero_state(batch_size, tf.float32)
        bot_final_state = cell_bot.zero_state(batch_size, tf.float32)
        bot_outputs = []
        with tf.variable_scope("bot_cell"):
            #  embedding
            with tf.device("cpu:0"):
                embedding = tf.get_variable(
                        "embedding", [vocab_size, size], dtype = tf.float32)
                word_vec = tf.nn.embedding_lookup(embedding, input_.targets_batch)
                print word_vec.shape
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                if time_step < frame_len:
                    bot_input = top_outputs[time_step]
                    (bot_output, bot_state_out) = cell_bot(bot_input, bot_state_in)
                    bot_state_in = bot_state_out
                else:
                    text_input = word_vec[:, time_step - 80, :]
                    top_cell_input = tf.reshape(top_outputs[time_step][0][size:], [1, size])
                    bot_input = tf.concat([text_input, top_cell_input],1)
                    (bot_output, bot_state_out) = cell_bot(bot_input, bot_state_in)
                    bot_state_in = bot_state_out
                    bot_final_state = bot_state_out
                    bot_outputs.append(bot_output)

        output = tf.reshape(tf.concat(axis = 1, values = bot_outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        if not is_training:
            return
        loss = tf.nn.sampled_softmax_loss(
                weights = tf.transpose(softmax_w), 
                biases = softmax_b, 
                labels = tf.reshape(input_.targets, [-1, 1]), 
                inputs = output, 
                num_sampled = 64,
                num_classes = vocab_size)

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = bot_final_state
        self._final_logits = logits
        self._final_probs = tf.nn.softmax(logits)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(0.01)
        self.train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step = tf.contrib.framework.get_or_create_global_step())

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state
    
    @property
    def final_logits(self):
        return self._final_logits

    @property
    def final_probs(self):
        return self._final_probs

def run_epoch(session, model, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
            iters * model.input.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)

f_data, t_data_raw = data_reader._read_data()
word_id = data_reader._build_word_id()
t_data_id, sent_len = data_reader._text_data_to_word_id(t_data_raw, word_id)
train_input = S2VT_input(
        frame_data = f_data, 
        text_data = t_data_id, 
        sent_len = sent_len, 
        word_id = word_id)
S2VT_model(is_training = True, input_ = train_input)
