import tensorflow as tf
import numpy as np
import data_reader
import time
import random
import bleu_eval as bleu
import re

class S2VT_input(object):
  def __init__(self, frame_data, text_data, sent_len, orig_sent_len, word_id, batch_size, name = None):
    self.frame_len = 80
    self.sent_len = sent_len
    self.vocab_size = len(word_id)
    self.batch_size = batch_size
    self.num_steps = self.frame_len + self.sent_len
    self.epoch_size = len(frame_data) // batch_size
    self.word_id = word_id
    self.input_batch, self.targets_batch, self.orig_sent_len = data_reader.Data_producer(
            frame_data = frame_data, 
            text_data = text_data, 
            batch_size = batch_size, 
            sent_len = sent_len,
            orig_sent_len = orig_sent_len)

class S2VT_model(object):
    def __init__(self, is_training, input_):
        self._input = input_
        size = 256
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(size, forget_bias = 0., state_is_tuple = True)
        frame_len = input_.frame_len
        sent_len = input_.sent_len
        self._orig_sent_len = input_.orig_sent_len[0]
        orig_sent_len = input_.orig_sent_len[0]
        vocab_size = input_.vocab_size
        batch_size = input_.batch_size
        num_steps = frame_len + sent_len
        # top
        cell_top = lstm_cell()
        #  if is_training:
            #  cell_top = tf.contrib.rnn.DropoutWrapper(cell_top, output_keep_prob = 0.5)
        input_frame = input_.input_batch
        top_state_in = cell_top.zero_state(batch_size, tf.float32)
        self._top_init_state = top_state_in
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
        #  if is_training:
            #  cell_bot = tf.contrib.rnn.DropoutWrapper(cell_bot, output_keep_prob = 0.5)
        bot_state_in = cell_bot.zero_state(batch_size, tf.float32)
        self._bot_init_state = bot_state_in
        bot_final_state = cell_bot.zero_state(batch_size, tf.float32)
        bot_inputs = []
        bot_outputs = []
        bot_output_word = []
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        with tf.variable_scope("bot_cell"):
            #  embedding
            with tf.device("cpu:0"):
                embedding = tf.get_variable(
                        "embedding", [vocab_size, size], dtype = tf.float32)
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                if time_step < frame_len:
                    bot_input = top_outputs[time_step]
                    (bot_output, bot_state_out) = cell_bot(bot_input, bot_state_in)
                    bot_state_in = bot_state_out
                else:
                    if time_step == frame_len:
                        embed_idx = [input_.word_id["BOS"] for _ in range(batch_size)]
                    else:
                        if is_training:
                            embed_idx = input_.targets_batch[:, time_step - 80 - 1]
                            word_idx = tf.argmax(bot_probs, 1)
                            bot_output_word.append(word_idx)
                        else:
                            embed_idx = tf.argmax(bot_probs, 1)
                            bot_output_word.append(embed_idx)

                    bot_inputs.append(embed_idx)
                    text_input = tf.nn.embedding_lookup(embedding, embed_idx)
                    top_cell_input = tf.reshape(top_outputs[time_step][:, size:], [batch_size, size])
                    bot_input = tf.concat([text_input, top_cell_input], 1)
                    (bot_output, bot_state_out) = cell_bot(bot_input, bot_state_in)
                    bot_logits = tf.matmul(bot_output, softmax_w) + softmax_b
                    bot_probs = tf.nn.softmax(bot_logits)
                    bot_state_in = bot_state_out
                    bot_final_state = bot_state_out
                    bot_outputs.append(bot_output)

        output = tf.reshape(tf.concat(axis = 1, values = bot_outputs), [-1, size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._bot_input_word = bot_inputs
        self._final_output_word = bot_output_word
        if not is_training:
            return
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(input_.targets_batch, [-1])],
                [tf.ones([batch_size * sent_len], dtype = tf.float32)])

        #  loss = tf.nn.sampled_softmax_loss(
                #  weights = tf.transpose(softmax_w), 
                #  biases = softmax_b, 
                #  labels = tf.reshape(input_.targets_batch[:,1:], [-1, 1]), 
                #  inputs = output, 
                #  num_sampled = 128,
                #  num_classes = vocab_size)

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = bot_final_state
        self._final_logits = logits
        self._final_probs = tf.nn.softmax(logits)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(0.001)
        #  self._train_op = optimizer.minimize(cost)
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step = tf.contrib.framework.get_or_create_global_step())

    @property
    def input(self):
        return self._input

    @property
    def orig_sent_len(self):
        return self._orig_sent_len

    @property
    def top_init_state(self):
        return self._top_init_state
    
    @property
    def bot_init_state(self):
        return self._bot_init_state

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def bot_input_word(self):
        return self._bot_input_word

    @property
    def final_state(self):
        return self._final_state
    
    @property
    def final_logits(self):
        return self._final_logits

    @property
    def final_probs(self):
        return self._final_probs

    @property
    def final_output_word(self):
        return self._final_output_word

def run_epoch(session, model, inv_word_id, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.
    iters = 0
    #  top_state = session.run(model.top_init_state)
    #  bot_state = session.run(model.bot_init_state)

    fetches = {
        "target_word": model.input.targets_batch,
        "cost": model.cost,
        "final_state": model.final_state,
        "final_word": model.final_output_word,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        tgt_word = vals["target_word"]
        pred_word = vals["final_word"]
        pred_word = np.asarray(pred_word)

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
        #  if verbose and step % 10 == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
            iters * model.input.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)

def run_predict(session, model, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.
    iters = 0
    fetches = {
        "input_frame": model._input.input_batch,
        "input_word": model.bot_input_word[1:],
        "target_word": model.input.targets_batch,
        "pred_word": model.final_output_word,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    feed_dict = {}
    vals = session.run(fetches, feed_dict)
    input_word = model.bot_input_word[0] + vals["input_word"]
    input_word = np.reshape(np.asarray(input_word), [-1])

    tgt_word = vals["target_word"]
    pred_word = vals["pred_word"]
    pred_word = np.asarray(pred_word)
    return input_word, tgt_word, pred_word

with tf.Graph().as_default():
    f_data, t_data_raw = data_reader._read_train_data()
    word_id, inv_word_id = data_reader._build_word_id()
    t_data_id, sent_len, orig_sent_len = data_reader._text_data_to_word_id(t_data_raw, word_id)

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.name_scope("Train"):
        train_input = S2VT_input(
                frame_data = f_data, 
                text_data = t_data_id, 
                sent_len = sent_len, 
                orig_sent_len = orig_sent_len,
                word_id = word_id,
                batch_size = 20,
                name = "train_input")
        with tf.variable_scope("model", reuse = False, initializer = initializer):
            train_model = S2VT_model(is_training = True, input_ = train_input)

    f_data, t_data_raw = data_reader._read_test_data()
    t_data_id, sent_len, orig_sent_len = data_reader._text_data_to_word_id(t_data_raw, word_id)
    with tf.name_scope("Test"):
        test_input = S2VT_input(
                frame_data = f_data, 
                text_data = t_data_id, 
                sent_len = sent_len, 
                orig_sent_len = orig_sent_len,
                word_id = word_id,
                batch_size = 1,
                name = "test_input")
        with tf.variable_scope("model", reuse = True, initializer = initializer):
            test_model = S2VT_model(is_training = False, input_ = test_input)

    sv = tf.train.Supervisor(logdir = None)
    saver = sv.saver
    with sv.managed_session() as session:
        for i in range(2000):
            cost = run_epoch(session = session, model = train_model, inv_word_id = inv_word_id, eval_op = train_model.train_op, verbose = True)
            print "Epoch %d, Cost %f"%(i, cost)
            if (i + 1) % 100 == 0:
                print "save model..."
                saver.save(session, './S2VT_model/S2VT_inputref', global_step = i + 1)
        #  saver.restore(session, './S2VT_model/S2VT-600')
                test_captions = data_reader._read_test_captions()
                bleu_score = 0.
                for j in range(test_input.epoch_size):
                    input_word, tgt_word, pred_word = run_predict(session = session, model = test_model, eval_op = None, verbose = False)
                    print "Testing %d"%j
                    print '-------------------------------------------------------'
                    pred_sent = ''
                    for w in pred_word[:, 0]:
                        if w != 0:
                            pred_sent += inv_word_id[w] + ' '
                    print "[predict sentence]: %s"%pred_sent
                    for tar_sent in test_captions[j]:
                        print "[target sentence] %s"%tar_sent
                        _bleu = 0.
                        pred_sent = re.sub('EOS', '', pred_sent)
                        tar_sent = re.sub('EOS', '', tar_sent)
                        _bleu += bleu.BLEU(pred_sent, tar_sent)
                        bleu_score += (_bleu / len(test_captions[j]))
                    print '-------------------------------------------------------'
                print "===== BLEU Score: [%.4f] =====\n"%(bleu_score / test_input.epoch_size)
