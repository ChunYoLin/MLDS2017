import data_reader
import numpy as np
import tensorflow as tf
import time

class lstm_input(object):
  def __init__(self, config, data, name = None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
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
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(input_.targets, [-1])],
                [tf.ones([batch_size * num_steps], dtype = tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._final_logits = logits
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
    
    return logits

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 1
  num_steps = 5
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 2
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10
  vocab_size = 12000

class TestConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 5
  hidden_size = 200
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10
  vocab_size = 12000

def main(_):
    f = './data/Holmes_Training_Data/14WOZ10.TXT'
    word_to_id = data_reader._build_vocab(f)
    inv_word_to_id = dict(zip(word_to_id.values(), word_to_id.keys()))
    train_data = data_reader._file_to_word_ids(f, word_to_id)
    test_all, test_before, test_after, test_answer = data_reader.test_data()
    test_data = []
    for line in test_before:
        test_data.append(data_reader._list_to_word_ids(line, word_to_id))
    config = SmallConfig()
    test_index = 10
    eval_config = TestConfig()
    eval_config.batch_size = 1
    test_len = len(test_data[test_index]) - 1
    print len(test_data[test_index])
    eval_config.num_steps = test_len
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.name_scope("Train"):
            train_input = lstm_input(config = config, data = train_data, name = "TrainInput")
            with tf.variable_scope("Model", reuse = False, initializer = initializer):
                m = lstm_model(is_training = True, config = config, input_ = train_input)

        with tf.name_scope("Test"):
            test_input = lstm_input(config = eval_config, data = test_data[test_index], name = "TestInput")
            with tf.variable_scope("Model", reuse = True, initializer = initializer):
                mtest = lstm_model(is_training = False, config = eval_config, input_ = test_input)


        sv = tf.train.Supervisor(logdir = "./model_checkpoints")
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.)
                m.assign_lr(session, config.learning_rate * lr_decay)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op = m._train_op, verbose = True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            ans = run_predict(session, mtest)
                

            print "test results"
            print test_data[test_index][: test_len]
            for option in test_answer[test_index]:
                if option in word_to_id:
                    print option, ans[test_len - 1][word_to_id[option]]
                else:
                    print option, 0
            

        
if __name__ == "__main__":
    tf.app.run()
