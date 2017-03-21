import data_reader
import numpy as np
import tensorflow as tf
import time
import csv
import os
import re
import cPickle as pk

class lstm_input(object):
  def __init__(self, config, data, name = None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    print self.epoch_size
    self.input_data, self.targets = data_reader.Data_producer(
        data, batch_size, num_steps, name = name)

class lstm_model():
    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 0., state_is_tuple = True)
        cell = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell() for _ in range(config.num_layers)], state_is_tuple = True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                    "embedding", [vocab_size, size], dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
            
        output = tf.reshape(tf.concat(axis = 1, values = outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        #  loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                #  [logits],
                #  [tf.reshape(input_.targets, [-1])],
                #  [tf.ones([batch_size * num_steps], dtype = tf.float32)])

        loss = tf.nn.sampled_softmax_loss(
                weights = tf.transpose(softmax_w), 
                biases = softmax_b, 
                labels = tf.reshape(input_.targets, [-1, 1]), 
                inputs = output, 
                num_sampled = 64,
                num_classes = vocab_size)

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._final_logits = logits
        self._final_probs = tf.nn.softmax(logits)
        if not is_training:
            return
        self._lr = tf.Variable(0., trainable = False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config. max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step = tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(
                tf.float32, shape = [], name = "new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

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
    
    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

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

def run_predict(session, model, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "final_state": model.final_state,
        "final_logits": model.final_logits,
        "final_probs": model.final_probs,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
    
    vals = session.run(fetches, feed_dict)
    state = vals["final_state"]
    logits = vals["final_logits"]
    probs = vals["final_probs"]

    return probs

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 5
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 12000

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 0
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 12000


def TEST_data():
    test_file = './data/testing_data.csv'
    with open(test_file)as f:
        test_reader = csv.reader(f, delimiter = ',')
        test_question_all = []
        test_question_before = []
        test_question_after = []
        test_answer = []
        max_len = 0
        for line in test_reader:
            for w_idx, w in enumerate(line):
                line[w_idx] = re.sub('\.', '', w)
            if line:
                state = 0
                line_before = []
                line_after = []
                for word in line[1].lower().split():
                    if word == '_____':
                        state = 1
                    elif state == 0:
                        line_before.append(word)
                    elif state == 1:
                        line_after.append(word)

                test_question_before.append(line_before)
                test_question_after.append(line_after)
                #  test_question_all.append(line[1])
                test_answer.append(line[2:])
    return test_question_before, test_question_after, test_answer

def TEST_data_to_id(word_to_id):

    test_before, test_after, test_answer = TEST_data()
    test_data_orig = []
    test_data_orig_len = []
    test_data_add_ans = []
    test_data_sync = []
    test_data_after = []
    max_len = 0
    #  create testing data, adding option into it
    for idx, line in enumerate(test_before):
        words = data_reader._list_to_word_ids(line, word_to_id)
        words_after = data_reader._list_to_word_ids(test_after[idx])
        test_data_orig.append(words)
        test_data_orig_len.append(len(words))
        words_a = []
        words_b = []
        words_c = []
        words_d = []
        words_e = []
        words_opt = [[] for i in range(5)]
        for opt, t_ans in enumerate(test_answer[idx]):
            if t_ans in word_to_id:
                words_opt[opt].extend(words + [word_to_id[t_ans]])
            else:
                words_opt[opt].extend(words + [0])

            if opt == 0:
                if t_ans in word_to_id:
                    words_a.extend(words + [word_to_id[t_ans]])
                else:
                    words_a.extend(words + [0])
                words_a.extend(words_after)
            elif opt == 1:
                if t_ans in word_to_id:
                    words_b.extend(words + [word_to_id[t_ans]])
                else:
                    words_b.extend(words + [0])
                words_b.extend(words_after)
            elif opt == 2:
                if t_ans in word_to_id:
                    words_c.extend(words + [word_to_id[t_ans]])
                else:
                    words_c.extend(words + [0])
                words_c.extend(words_after)
            elif opt == 3:
                if t_ans in word_to_id:
                    words_d.extend(words + [word_to_id[t_ans]])
                else:
                    words_d.extend(words + [0])
                words_d.extend(words_after)
            elif opt == 4:
                if t_ans in word_to_id:
                    words_e.extend(words + [word_to_id[t_ans]])
                else:
                    words_e.extend(words + [0])
                words_e.extend(words_after)
        test_data_add_ans.append(words_a)
        test_data_add_ans.append(words_b)
        test_data_add_ans.append(words_c)
        test_data_add_ans.append(words_d)
        test_data_add_ans.append(words_e)

    #  make every test data same length with zeros
    for line in test_data_add_ans:
        for i in range(max_len - len(line) + 1):
            line.append(0)
        for w in line:
            test_data_sync.append(w)
    test_data_sync.append(0)

    #  create a list contain the first word after the ______ in test sentence
    for idx, line in enumerate(test_after):
        word = data_reader._list_to_word_ids(line, word_to_id)
        if word:
            test_data_after.append(word[0])
        else:
            test_data_after.append(-1)

    return test_data_orig_len, test_data_sync, test_after, test_answer

def main(_):
    train_config = MediumConfig()
    test_config = MediumConfig()
    test_config.batch_size = 1
    
    f_list = []
    for file_id, filename in enumerate(os.listdir('./data/Holmes_Training_Data')):
        if file_id < 500:
            filename = './data/Holmes_Training_Data/' + filename
            f_list.append(filename)
    #  word_to_id = data_reader._build_multi_vocab(f_list, train_config.vocab_size)
    #  inv_word_to_id = dict(zip(word_to_id.values(), word_to_id.keys()))
    with open('w2id_500P_12000V.pk', 'rb') as pickle_file:
        word_to_id = pk.load(pickle_file)
        inv_word_to_id = dict(zip(word_to_id.values(), word_to_id.keys()))
    #  train_data = data_reader._multi_file_to_word_ids(f_list, word_to_id)
    with open('train_data_500P_12000V.pk', 'rb') as pickle_file:
        train_data = pk.load(pickle_file)
    print "finish loading training data"

    test_data_orig_len, test_data_sync, test_data_after, test_answer = TEST_data_to_id(word_to_id)
    #  max_len = max(test_data_orig_len)
    #  test_config.num_steps = max_len + 1
    #  print "finish processing testing data"

    #  with tf.Graph().as_default():
        #  initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
        #  with tf.name_scope("Train"):
            #  train_input = lstm_input(config = train_config, data = train_data, name = "TrainInput")
            #  with tf.variable_scope("Model", reuse = False, initializer = initializer):
                #  m = lstm_model(is_training = True, config = train_config, input_ = train_input)

        #  with tf.name_scope("Test"):
            #  test_input = lstm_input(config = test_config, data = test_data_sync, name = "TestInput")
            #  with tf.variable_scope("Model", reuse = True, initializer = initializer):
                #  mtest = lstm_model(is_training = False, config = test_config, input_ = test_input)

        #  sv = tf.train.Supervisor(logdir = "./model_checkpoints")
        #  with sv.managed_session() as session:
            #  for i in range(train_config.max_max_epoch):
                #  lr_decay = train_config.lr_decay ** max(i + 1 - train_config.max_epoch, 0.)
                #  m.assign_lr(session, train_config.learning_rate * lr_decay)
                #  print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                #  train_perplexity = run_epoch(session, m, eval_op = m._train_op, verbose = True)
                #  print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            #  probs = []
            #  ans = np.zeros([mtest._input.epoch_size / 5, 5], np.float32)
            #  for i in range(mtest._input.epoch_size / 5):
                #  print("processing %d/%d"%((i + 1) * 5, mtest._input.epoch_size))
                #  probs_option = []
                #  for j in range(5):
                    #  probs_option.append(run_predict(session, mtest))
                #  probs.append(probs_option)
            
            #  for idx, orig_len in enumerate(test_data_orig_len):
                #  for option_id, option in enumerate(test_answer[idx]):
                    #  if option in word_to_id:
                        #  prob_before = probs[idx][option_id][orig_len - 1, word_to_id[option]]
                        #  word_after = test_data_after[idx]
                        #  prob_after = probs[idx][option_id][orig_len, word_after]
                        #  ans[idx, option_id] = prob_before * prob_after
                    #  else:
                        #  ans[idx, option_id] = 0.
            #  with open('ans.csv', 'w') as f:
                #  f.write("id,answer\n")
                #  for idx, out in enumerate(ans):
                    #  if idx > 0:
                        #  if np.argmax(ans[idx]) == 0:
                            #  f.write(str(idx) + ',a\n')
                        #  elif np.argmax(ans[idx]) == 1:
                            #  f.write(str(idx) + ',b\n')
                        #  elif np.argmax(ans[idx]) == 2:
                            #  f.write(str(idx) + ',c\n')
                        #  elif np.argmax(ans[idx]) == 3:
                            #  f.write(str(idx) + ',d\n')
                        #  elif np.argmax(ans[idx]) == 4:
                            #  f.write(str(idx) + ',a\n')
                
        
if __name__ == "__main__":
    tf.app.run()
