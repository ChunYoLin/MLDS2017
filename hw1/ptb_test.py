import time
import os
import re
import collections
import csv
import numpy as np
import tensorflow as tf

#from gensim.models import Word2Vec
'''
print "Loading model"
model = Word2Vec.load('mymodel')


print model.similar_by_word('million')
print model.similar_by_word('information')
'''


flags = tf.flags
logging = tf.logging

# Flags : TF's command line module, ==> (--model, default, help description)

flags.DEFINE_string(
    "model", "medium",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("train", False,
                  "Training model or Loading model")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

def read_data(dirname):
    words = []
    i=0
    for filename in os.listdir(dirname):
        i+=1
        if i > 101:
            break
        for line in open(os.path.join(dirname, filename)):
            words.extend(re.sub('[^A-Za-z ]','',line).split())

    return words

def read_test_data(filename):
    opt = ['a', 'b', 'c', 'd', 'e']
    #to_num = lambda word: dic.get(word, 0)
    data = []
    opt_words = []
    F = open(filename, 'r')
    reader = csv.reader(F)
    next(reader, None)
    max_len = 0
    for row in reader:
        que = {}
        que['id'] = row[0]
        _sent = re.sub('[^A-Za-z_ ]','',row[1]).split()
        _pos = _sent.index('_____')
        _sent = _sent[0:_pos+1]
        #que['sentence'] = map(to_num, _sent)
        que['sentence'] = _sent
        if len(que['sentence']) > max_len:
            max_len = len(que['sentence'])

        for i in range(5):
            que[ opt[i] ] = row[i+2]
            if row[i+2] not in opt_words:
                opt_words.append(row[i+2])
        data.append(que)
    F.close
    return data, opt_words, max_len

def build_dataset(words, test_que, opt_words, config):
    count = [['UNK', -1]]
    # Counter can calculate the occur num of word in words list.
    # most_common will pick the most common occur word.
    # Result : [('ab', 24), ('bc', 12), ...]
    c = collections.Counter(words)
    count.extend(c.most_common(config.vocab_size - 1))
    print "Total Vocab Size %d" % len(c)
    _com_words = [word for word, _ in count]
    _num = count[-1][1]
    for _w in opt_words:
        if _w not in _com_words:
            count.append((_w, c[_w]))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))


    to_num = lambda word: dictionary.get(word, 0)
    to_word = lambda ind: reverse_dictionary[ind]
    for que in test_que:
        que['sentence'] = map(to_num, que['sentence'])
    for i in range(2):
        print test_que[i]['sentence']
        print map(to_word, test_que[i]['sentence'])
    test_size = len(test_que)
    num = config.num_steps
    test_data = np.zeros( (test_size*num)+1 )
    for i in range(test_size):
        _len = len(test_que[i]['sentence'])
        test_data[ i*num : i*num+_len ] = test_que[i]['sentence']

    return data, test_data, count, dictionary, reverse_dictionary
    # Data : Like actual word list, but store the word's index in count
    # Count : store the word and its occur numbers. from biggest to smallest
    # dictionary : store the word and its word's index in count

def data_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "DataProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                                            [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                                                 [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                                                 [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

class Input(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = data_producer(
                data, batch_size, num_steps, name=name)
        # Input_data is raw data with shape [BatchSize, num_steps(slide widow)]
        # Example " I like you, I like she " (Batch=2, step=2)
        # Input:[[I, like], [I, like]] taget:[[like, you], [like, she]]
        # So, one batch can parallel training.

class RNNModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_
        self._x = input_.input_data
        self._y = input_.targets

        batch_size = input_.batch_size
        num_steps = input_.num_steps    # The number of RNN Node with expanding
        size = config.hidden_size       # The number of node in each RNN node.
        vocab_size = config.vocab_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        # Create lstm cell with size=[Size]
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True)
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                        lstm_cell(), output_keep_prob=config.keep_prob)

        # Create multi layer RNN Cell, each cell with [Size]
        cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        # Build word embedding, the vector length of word is [Size]
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                    "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.nn.rnn(cell, inputs,
        #                              initial_state=self._initial_state)
        '''
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # inputs : [BatchSize, Numsteps, WordEmbedding]
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                # cell_output : [BatchSize, Size]
                # outputs : [Numsteps, BatchSize, Size]
                outputs.append(cell_output)
        '''
        #inputs = tf.unstack(inputs, num=num_steps, axis=1)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state = self._initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        # output : [Numsteps * BatchSize, Size]
        softmax_w = tf.get_variable(
                "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        '''
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(input_.targets, [-1])],
                [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        '''
        #'''
        loss = tf.nn.sampled_softmax_loss(
                                weights = tf.transpose(softmax_w),
                                biases = softmax_b,
                                inputs = output,
                                labels = tf.reshape(input_.targets, [-1, 1]),
                                num_sampled = 64,
                                num_classes = vocab_size )
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        #'''
        self._final_state = state
        self._probs = tf.nn.softmax(logits)

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                    config.max_grad_norm)
        #'''
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())
        #'''
        #optimizer = tf.train.AdamOptimizer(self._lr)
        #self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
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
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def probs(self):
        return self._probs

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 30
    hidden_size = 256
    max_epoch = 2
    max_max_epoch = 2
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
    num_steps = 30
    hidden_size = 512
    max_epoch = 6
    max_max_epoch = 10
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 12000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
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

def run_predict(session, model):

    state = session.run(model.initial_state)
    fetches = {
            "x": model.x,
            "y": model.y,
            "probs": model.probs,
    }
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    x = vals["x"]
    y = vals["y"]
    probs = vals["probs"]
    x = np.reshape(x, (-1))
    y = np.reshape(y, (-1))
    return x, y, probs

def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def max_prob_word(probs, que, dic):
    opt = ['a', 'b', 'c', 'd', 'e']
    prob = lambda ind: probs[ dic.get(que[ind], 0) ]
    opt_prob = map(prob, opt)
    #print ans_prob
    ind = opt[ opt_prob.index(max(opt_prob)) ]
    #print que[ind]
    return ind


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    print "=== Test Data ==="
    test_que, opt_words, sent_len = read_test_data("testing_data.csv")
    print "Max Sent Len : %d" % sent_len
    print "Opt words num : %d" % len(opt_words)

    print "=== Train Data ==="
    words = read_data(FLAGS.data_path)
    valid_size = len(words) / 20
    print "Data Size ", len(words)
    print "Valid Size ", valid_size
    config = get_config()
    data, test_data, count, dictionary, reverse_dictionary = \
                    build_dataset(words, test_que, opt_words, config)
    del words  # Hint to reduce memory
    to_word = lambda ind: reverse_dictionary[ind]
    config.vocab_size = len(count)
    print "vocab_size %d" % len(count)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    # data is a list of word's id, it translate word to id, like '<UNK>' : 1.

    valid_data = data[0:valid_size]
    train_data = data[valid_size:-1]
    #'''

    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.vocab_size = config.vocab_size

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
            config.init_scale)

        with tf.name_scope("Train"):
            train_input = Input(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = RNNModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = Input(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = RNNModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)
        with tf.name_scope("Test"):
            test_input = Input(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = RNNModel(is_training=False, config=eval_config, input_=test_input)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        saver=sv.saver
        with sv.managed_session() as session:
            if FLAGS.train:
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    #lr_decay = 0.002 * (0.97**i)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity = run_epoch(session, m, 
                                            eval_op=m.train_op, verbose=True)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1,
                                            train_perplexity))
                    valid_perplexity = run_epoch(session, mvalid)
                    print("Epoch: %d Validation Perplexity: %.3f" % (i + 1,
                                            valid_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print "Validation Perplexity: %.3f" % valid_perplexity
            F = open("result.csv", "wb")
            writer = csv.writer(F, delimiter=',')
            row = []
            row.append('id')
            row.append('answer')
            writer.writerow(row)
            num = 0
            for que in test_que:
                row = []
                x, y, probs = run_predict(session, mvalid)
                pred_word = np.argmax(probs, axis=1)
                space_ind = len(que['sentence'])
                num += 1
                if num < 4:
                    print "==== %d ====" % num
                    print [to_word(x[i]) for i in range(config.num_steps)]
                    print [to_word(pred_word[i]) for i in range(config.num_steps) ]
                    print "Index : %d" % space_ind
                answer = 'c'
                if space_ind - 2 > 0:
                    answer = max_prob_word(probs[space_ind-2,:], que, dictionary)
                row.append(que['id'])
                row.append(answer)
                writer.writerow(row)
            F.close()

            if FLAGS.save_path and FLAGS.train:
                print("Saving model to %s." % FLAGS.save_path)
                saver.save(session, FLAGS.save_path, global_step=sv.global_step)
    #'''
if __name__ == "__main__":
    tf.app.run()
