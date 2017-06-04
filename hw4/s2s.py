import os
import re
import json
import collections
import numpy as np
import time
import tensorflow as tf
import inspect
import bleu_eval as bleu
import nltk
import math

TRAINING_DATA = "Data/train.txt"
TESTING_DATA = "Data/test.txt"
VALID_NUM = 5
LIMIT_LEN = 30

flags = tf.flags
logging = tf.logging

# Flags : TF's command line module, ==> (--model, default, help description)

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("train", False,
                  "Training model or Loading model")
flags.DEFINE_bool("sample", True,
                  "Use Sample Scheduling")
flags.DEFINE_string("out", None,
                  "Write out file")

FLAGS = flags.FLAGS


def inverse_sigmoid_decay(x):
    return 1.0 / (1.0+math.exp(8*(x-0.6)))

def data_type(flag):
    if flag == 0:
        return tf.float32# if FLAGS.use_fp16 else tf.float32
    elif flag == 1:
        return np.float32

def find_pad(sent):
    try:
        pos = np.where(sent == 2)[0][0]
    except IndexError:
        pos = 3
    return pos

def parse_sent(_sent):
    _sent = re.sub("[^A-Za-z ]", '', _sent)
    _sent = _sent.lower()
    return _sent.split()

def build_words_dic(words, config):
    count = [['UNK', -1], ['_BOS', -1], ['_PAD', -1]]
    # Counter can calculate the occur num of word in words list.
    # most_common will pick the most common occur word.
    # Result : [('ab', 24), ('bc', 12), ...]
    c = collections.Counter(words)
    count.extend(c.most_common(config.vocab_size - 3))
    print "Raw Vocab Size %d" % len(c)
    print "Total Vocab Size %d" % len(count)

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reverse_dictionary
    # dictionary : store the word and its word's index in count
    # to_num = lambda word: dictionary.get(word, 0)
    # to_word = lambda ind: reverse_dictionary[ind]

def read_file():
    data_file = TRAINING_DATA
    test_file = TESTING_DATA
    total_words = []
    max_len = 0
    input_raw_list = []
    target_raw_list = []
    test_raw_list = []
    with open(data_file, 'r') as F:
        all_lines = F.read().splitlines()
        for i in range(0, len(all_lines), 2):
            _x = nltk.word_tokenize(all_lines[i].lower())
            _y = nltk.word_tokenize(all_lines[i+1].lower())
            if len(_x) > LIMIT_LEN or len(_y) > LIMIT_LEN:
                continue
            input_raw_list.append(_x)
            target_raw_list.append(_y)
            total_words.extend(_x)
            total_words.extend(_y)
            max_len = max(max_len, len(_x), len(_y))

    with open(test_file, 'r') as F:
        all_lines = F.read().splitlines()
        for line in all_lines:
            _x = nltk.word_tokenize(line.lower())
            test_raw_list.append(_x)
    return input_raw_list, target_raw_list, test_raw_list, total_words, max_len+2

def build_dataset(input_raw_list, target_raw_list, words_dic, config):
    to_num = lambda word: words_dic.get(word, 0)

    data_size = config.data_size
    sent_len = config.sent_len
    _input = np.full((data_size, sent_len), 2, dtype=np.int)
    _target = np.full((data_size, sent_len), 2, dtype=np.int)
    for i in range(data_size):
        _x = map(to_num,  input_raw_list[i])
        _input[i, 0:len(_x)] = _x
        _target[i,0] = 1
        if target_raw_list != None:
            _y = map(to_num,  target_raw_list[i])
            _target[i,1:len(_y)+1] = _y
    return [_input, _target], \
            [_input[0:VALID_NUM, :], _target[0:VALID_NUM, :]]

def data_producer(data, data_size, sent_len, \
                        batch_size, shuffle, name=None):
    with tf.name_scope(name, "DataProducer", \
            [data, data_size, sent_len, batch_size, shuffle]):

        x_data = data[0]
        y_data = data[1]

        i = tf.train.range_input_producer(data_size, shuffle=shuffle).dequeue()
        x_tensor = tf.convert_to_tensor(x_data, name="x_data", \
                                                        dtype=tf.int32)
        x = tf.strided_slice(x_tensor, [i, 0], [(i+1), sent_len])
        x = tf.reshape(x, [sent_len])
        x.set_shape([sent_len])

        y_tensor = tf.convert_to_tensor(y_data, name="y_data", \
                                                        dtype=tf.int32)
        y = tf.strided_slice(y_tensor, [i, 0], [(i+1), sent_len])
        y = tf.reshape(y, [sent_len])
        y.set_shape([sent_len])

        x_batch, y_batch = tf.train.batch(
                                [x, y],
                                batch_size = batch_size)
        return x_batch, y_batch

class Input(object):
    """The input data."""

    def __init__(self, config, data, shuffle, name=None):
        self.batch_size = batch_size = config.batch_size
        self.data_size = data_size = config.data_size
        self.sent_len = sent_len = config.sent_len
        self.num_steps = 2 * sent_len
        self.epoch_size = int(data_size // batch_size)
        #self.epoch_size = int(data_size)
        self.inputs, self.targets = data_producer(
                data, data_size, sent_len, \
                batch_size, shuffle, name=name)
        print self.inputs.shape
        print self.targets.shape
        # Input_data = Ask Sent,  Target = Response Sent
        # With producer, each run epoch will produce one batch
        # Input_data = [Batch_Size, Sent_len], Target = [Batch_size, Sent_len]

class Seq2SeqModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_
        self._x = input_.inputs
        self._y = input_.targets
        batch_size = input_.batch_size
        num_steps = input_.num_steps    # The number of RNN Node with expanding
        size = config.hidden_size       # The number of node in each RNN node.
        vocab_size = config.vocab_size
        sent_len = input_.sent_len
        self._steps = steps = 0

        self._sample_prob = tf.Variable(1.0, trainable=False)
        self._new_sample_prob = tf.placeholder( \
                        tf.float32, shape=[], name="new_sample_prob")
        self._sample_prob_update = tf.assign( \
                        self._sample_prob, self._new_sample_prob)

        embedding = tf.get_variable("embedding", [vocab_size, size], \
                                            dtype=data_type(0))
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type(0))
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type(0))

        def lstm_cell():
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is unfortunately not
        # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
        # an argument check here:
            if 'reuse' in \
                    inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                            size, forget_bias=1.0, state_is_tuple=True,
                                        reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                              size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                        lstm_cell(), output_keep_prob=config.keep_prob)

        if config.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(
                    [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        else:
            cell  = attn_cell()
        inputs = tf.nn.embedding_lookup(embedding, input_.inputs)
        targets = tf.nn.embedding_lookup(embedding, input_.targets)

        self._initial_state = cell.zero_state(batch_size, data_type(0))

        #'''
        outputs = []
        with tf.variable_scope("Seq2Seq"):
            # Encoder Stage:
            state = self._initial_state
            for time_step in range(sent_len):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                _, state = cell(inputs[:, time_step, :], state)
            #_, state = tf.nn.dynamic_rnn(cell, inputs, \
            #                initial_state = self._initial_state)
            # Decoder Stage:
            last_word_index = tf.zeros((batch_size), dtype=tf.int32)
            for time_step in range(sent_len-1):

                def target_input(): return targets[:, time_step, :]
                def predict_input(): \
                        return tf.nn.embedding_lookup(embedding, last_word_index)
                if time_step > 0:
                    if not is_training:
                        response = predict_input()
                    elif FLAGS.sample:
                        _rand = tf.random_uniform([1])[0]
                        response = tf.cond(_rand <= self._sample_prob, \
                                    target_input, predict_input)
                    else:
                        response = target_input()
                    response = target_input()
                else:
                    response = target_input()

                # Multi Layer RNN
                tf.get_variable_scope().reuse_variables()
                (cell_output, state) = \
                                cell(response, state)
                _last_output = tf.reshape(cell_output, [-1, size])
                _last_logits = tf.matmul(_last_output, softmax_w) + softmax_b
                _last_probs = tf.nn.softmax(_last_logits)
                last_word_index = tf.argmax(_last_probs, axis=1)
                # Only get Decoding stage Output.
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        # output : [ (SentLen-1) * BatchSize, Size]
        logits = tf.matmul(output, softmax_w) + softmax_b
        #'''
        '''
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                                [logits],
                                #[tf.reshape(input_.targets[1:steps[0]], [-1])],
                                [tf.reshape(input_.targets[:, 1:], [-1])],
                                #[tf.ones([steps[0]-1], dtype=data_type(0))])
                                [tf.ones([batch_size*(sent_len-1)], dtype=data_type(0))])
        #'''
        #'''
        loss = tf.nn.sampled_softmax_loss(
        #loss = tf.nn.nce_loss(
                                weights = tf.transpose(softmax_w),
                                biases = softmax_b,
                                inputs = output,
                                labels = tf.reshape( \
                                        input_.targets[:, 1:], [-1, 1]),
                                num_sampled = 64,
                                num_classes = vocab_size )
        #'''
        #'''
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._probs = tf.nn.softmax(logits)
        if not is_training:
            return
        self._train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
        #'''
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_sample(self, session, sample_value):
        session.run(self._sample_prob_update, \
            feed_dict={self._new_sample_prob: sample_value})

    @property
    def input(self):
        return self._input

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def steps(self):
        return self._steps

    @property
    def init_first_state(self):
        return self._init_first_state

    @property
    def init_second_state(self):
        return self._init_second_state

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def probs(self):
        return self._probs

    @property
    def sample_prob(self):
        return self._sample_prob

    @property
    def train_op(self):
        return self._train_op

def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    costs = 0.0
    iters = 0
    fetches = {
            "cost": model.cost,
            "sample": model.sample_prob
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        sample = vals["sample"]

        costs += cost
        iters += 1

    return (costs / iters), sample

def run_predict(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    fetches = {
            "x": model.x,
            "y": model.y,
            "probs": model.probs,
    }
    feed_dict = {}
    vals = session.run(fetches, feed_dict)
    x = np.reshape(vals["x"], (-1))
    y = np.reshape(vals["y"], (-1))
    probs = vals["probs"]

    pred_words = np.argmax(probs, axis=1)
    _pos_x = find_pad(x)
    _pos_y = find_pad(y)
    _pos_pred = find_pad(pred_words)
    x = x[0:_pos_x+1]
    y = y[1:_pos_y]
    pred = pred_words[0:_pos_pred]
    return x.tolist(), y.tolist(), pred.tolist()

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 0
    hidden_size = 256
    max_epoch = 6
    max_max_epoch = 200
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 200
    vocab_size = 3000
    data_size = 0
    sent_len = 0

def main(_):
    config = MediumConfig()
    valid_config = MediumConfig()
    eval_config = MediumConfig()
    print "\n--- Read Data ---"
    train_raw_input, train_raw_target, test_raw_input, total_words, sent_len = read_file()
    print "\n--- Build Words Dict ---"
    words_dic, rev_words_dic = build_words_dic(total_words, config)
    del total_words

    to_word = lambda ind: rev_words_dic[ind]
    to_num = lambda word: words_dic.get(word, 0)
    train_data_size = len(train_raw_input)
    test_data_size = len(test_raw_input)
    print "Train Data size ",train_data_size
    print "Test Data size ",test_data_size
    print "Train max sent len ",sent_len
    config.data_size = train_data_size
    config.sent_len = sent_len
    valid_config.data_size = VALID_NUM
    valid_config.sent_len = sent_len
    valid_config.batch_size = 1
    eval_config.data_size = test_data_size
    eval_config.sent_len = sent_len
    eval_config.batch_size = 1
    train_data, valid_data = build_dataset(train_raw_input, train_raw_target, \
                                               words_dic, config)
    test_data, _ = build_dataset(test_raw_input, None, \
                                               words_dic, eval_config)

    del train_raw_input
    del train_raw_target
    #'''
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
            config.init_scale)
        print "\n--- Build Train Model ---"
        with tf.name_scope("Train"):
            train_input = Input(config=config, data=train_data, \
                                    shuffle=True, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Seq2SeqModel(is_training=True, config=config, input_=train_input)
        print "\n--- Build Valid Model ---"
        with tf.name_scope("Valid"):
            valid_input = Input(config=valid_config, data=valid_data, \
                                    shuffle=False, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Seq2SeqModel(is_training=False, config=valid_config, \
                                                            input_=valid_input)
        print "\n--- Build Test Model ---"
        with tf.name_scope("Test"):
            test_input = Input(config=eval_config, data=test_data, \
                                    shuffle=False, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = Seq2SeqModel(is_training=False, config=eval_config, \
                                                            input_=test_input)
        #sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        sv = tf.train.Supervisor(logdir=None)
        saver=sv.saver
        start_time = time.time()
        with sv.managed_session() as session:

            if FLAGS.train:
                for epoch_step in range(config.max_max_epoch):
                    _percent = (float(epoch_step) / config.max_max_epoch)
                    #sample_decay = 1.0 - _percent
                    sample_decay = inverse_sigmoid_decay(_percent)
                    m.assign_sample(session, sample_decay)
                    perplexity, sample = run_epoch(session, m, \
                                            eval_op=m.train_op, verbose=True)
                    if ((epoch_step+1) % 20) == 0:
                        run_time = int( (time.time() - start_time) / 60 )
                        print("Epoch %d Perplexity %.3f Sample %.2f  Run %d min" \
                                    % ((epoch_step+1), perplexity, sample, run_time))
                    if ((epoch_step+1) % 100) == 0:
                        #bleu_score = 0.0
                        print "===== Epoch %d Test Data =====" % (epoch_step+1)
                        print "===== Test Data ====="
                        for i in range(eval_config.data_size):
                            x, y, pred = run_predict(session, mtest)
                            cand_sent = ' '.join( map(to_word, pred) )
                            x_sent = ' '.join( map(to_word, x) )
                            y_sent = ' '.join( map(to_word, y) )
                            print "In: %s\nRe: %s\nPr: %s\n" % (x_sent, y_sent, cand_sent)

            if FLAGS.save_path and FLAGS.train:
                print("Saving model to %s." % FLAGS.save_path)
                saver.save(session, FLAGS.save_path, global_step=sv.global_step)
    print "END"
    #'''
if __name__ == "__main__":
    tf.app.run()






