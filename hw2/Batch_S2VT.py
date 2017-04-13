import os
import re
import json
import collections
import numpy as np
import time
import tensorflow as tf
import inspect

TRAIN_PATH = "Data/training_data/feat/"
TRAIN_LABEL = "Data/training_label.json"
TEST_PATH = "Data/testing_data/feat/"
TEST_LABEL = "Data/testing_public_label.json"

def data_type(flag):
    if flag == 0:
        return tf.float32# if FLAGS.use_fp16 else tf.float32
    elif flag == 1:
        return np.float32

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

def read_file(config, is_train):
    if is_train:
        label = TRAIN_LABEL
        path = TRAIN_PATH
    else:
        label = TEST_LABEL
        path = TEST_PATH
    data = {}
    total_words = []
    max_len = 0
    with open(label, 'r') as F:
        label_data = json.load(F)
        for _dic in label_data:
            if _dic['id'] in data:
                print "[Error] Label Data id Already in data"
            _x = {}
            _x['sent'] = []
            sent_list = _dic['caption']

            for sent in sent_list:
                words = parse_sent(sent)
                if len(words) > max_len:
                    max_len = len(words)
                _x['sent'].append(words)
                total_words.extend(words)

            filename = _dic['id'] + '.npy'
            _feat = np.load( path + filename )
            _x['feat'] = _feat.astype( data_type(1) )
            data[_dic['id']] = _x

        del label_data
    print "--- Read Label ---"
    print "Data Len: ",len(data)
    print "Total Words num: ",len(total_words)
    print "Max sent len: ",max_len
    return data, total_words, max_len+2

def build_dataset(data, words_dic, config):
    to_num = lambda word: words_dic.get(word, 0)

    data_size = config.data_size
    sent_len = config.sent_len
    frame_num = config.frame_num
    feat_size = config.feat_size
    _input = np.zeros((data_size, frame_num, feat_size), dtype=data_type(1))
    _target = np.full((data_size, sent_len), 2, dtype=np.int)
    _step_list = np.zeros((data_size), dtype = np.int)
    i=0
    for key, _dic in data.iteritems():
        _input[i,:,:] = _dic['feat']
        _list = map(to_num,  _dic['sent'][0])
        _target[i,0] = 1
        _target[i,1:len(_list)+1] = _list
        _step_list[i] = len(_list)+2
        i+=1
    return [_input, _target, _step_list], \
                    [_input[0:3, :], _target[0:3, :], _step_list[0:3]]

def data_producer(data, data_size, sent_len, \
                        frame_num, feat_size, batch_size, shuffle, name=None):
    with tf.name_scope(name, "DataProducer", \
            [data, data_size, sent_len, frame_num, feat_size, batch_size, shuffle]):

        x_data = data[0]
        y_data = data[1]
        steps_data = data[2]

        i = tf.train.range_input_producer(data_size, shuffle=shuffle).dequeue()
        x_tensor = tf.convert_to_tensor(x_data, name="x_data", \
                                                        dtype=data_type(0))
        x = tf.strided_slice(x_tensor, [i, 0, 0], [(i+1), frame_num, feat_size])
        x = tf.reshape(x, [frame_num, feat_size])
        x.set_shape([frame_num, feat_size])

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
        self.frame_num = frame_num = config.frame_num
        self.feat_size = feat_size = config.feat_size
        self.data_size = data_size = config.data_size
        self.sent_len = sent_len = config.sent_len
        self.num_steps = frame_num + sent_len
        self.epoch_size = int(data_size // batch_size)
        #self.epoch_size = int(data_size)
        self.input_data, self.targets = data_producer(
                data, data_size, sent_len, frame_num, feat_size, \
                                        batch_size, shuffle, name=name)
        print self.input_data.shape
        print self.targets.shape
        # Input_data = video frame feature,  Target = Video decription ( 1 sentence )
        # With producer, each run epoch will produce one data 
        # Input_data = [Batch_Size, 80, 4096], Target = [Batch_size, Sent_len]

class S2VTModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_
        self._x = input_.input_data
        self._y = input_.targets
        batch_size = input_.batch_size
        num_steps = input_.num_steps    # The number of RNN Node with expanding
        size = config.hidden_size       # The number of node in each RNN node.
        vocab_size = config.vocab_size
        frame_num = input_.frame_num
        feat_size = input_.feat_size
        sent_len = input_.sent_len
        self._steps = steps = 0

        frame_padding = tf.zeros([batch_size, sent_len, size], tf.float32)
        text_padding = tf.zeros([batch_size, frame_num, size/2], data_type(0))

        weight_1 = tf.get_variable("weight_1", [feat_size, size], \
                                            dtype=data_type(0))
        weight_2 = tf.get_variable("weight_2", [size, size/2], \
                                            dtype=data_type(0))
        embedding = tf.get_variable("embedding", [vocab_size, size/2], \
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

        first_cell = second_cell = attn_cell()
        #second_cell = attn_cell()
        self._init_first_state = first_cell.zero_state(batch_size, data_type(0))
        self._init_second_state = second_cell.zero_state(batch_size, data_type(0))
        first_state = self._init_first_state
        second_state = self._init_second_state

        targets = tf.nn.embedding_lookup(embedding, input_.targets)
        inputs = tf.reshape(input_.input_data, (-1, feat_size))
        inputs = tf.matmul(inputs, weight_1)
        inputs = tf.reshape(inputs, (batch_size, frame_num, size))
        frame_inputs = tf.concat([inputs, frame_padding], axis = 1)
        text_inputs = tf.concat([text_padding, targets], axis = 1)
        print "Frame Input shape: ",frame_inputs.shape
        print "Text Input shape: ",text_inputs.shape
        #'''
        outputs = []
        with tf.variable_scope("S2VT"):
            last_word_index = tf.zeros((batch_size), dtype=tf.int32)
            for time_step in range(num_steps-1):
                # First Layer RNN
                if time_step > 0 : tf.get_variable_scope().reuse_variables()
                (first_output, first_state) = \
                        first_cell(frame_inputs[:, time_step, :], first_state)
                second_input_1 = tf.matmul(first_output, weight_2)
                #second_input_2 = tf.reshape( \
                #            text_inputs[:, time_step, :], [1, size/2])
                second_input_2 = text_inputs[:, time_step, :]
                if (not is_training) and time_step > frame_num:
                    second_input_2 = tf.nn.embedding_lookup(embedding, last_word_index)
                second_input = tf.concat([second_input_1, second_input_2], axis=1)

                # Second Layer RNN
                tf.get_variable_scope().reuse_variables()
                (second_output, second_state) = \
                                second_cell(second_input, second_state)
                _last_output = tf.reshape(second_output, [-1, size])
                _last_logits = tf.matmul(_last_output, softmax_w) + softmax_b
                _last_probs = tf.nn.softmax(_last_logits)
                last_word_index = tf.argmax(_last_probs, axis=1)
                # Only get Decoding stage Output.
                if time_step >= frame_num:
                    outputs.append(second_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        print "output shape: ",output.shape
        # output : [ (SentLen-1) * BatchSize, Size]
        logits = tf.matmul(output, softmax_w) + softmax_b
        #'''
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                                [logits],
                                #[tf.reshape(input_.targets[1:steps[0]], [-1])],
                                [tf.reshape(input_.targets[:, 1:], [-1])],
                                #[tf.ones([steps[0]-1], dtype=data_type(0))])
                                [tf.ones([batch_size*(sent_len-1)], dtype=data_type(0))])
        #'''
        '''
        loss = tf.nn.sampled_softmax_loss(
                                weights = tf.transpose(softmax_w),
                                biases = softmax_b,
                                inputs = output,
                                labels = tf.reshape( \
                                        input_.targets[:, 1:], [-1, 1]),
                                num_sampled = 64,
                                num_classes = vocab_size )
        #'''
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._probs = tf.nn.softmax(logits)
        if not is_training:
            return
        self._train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
        '''
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                    config.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())
        #optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        #self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        #'''
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

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
    def train_op(self):
        return self._train_op

def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    costs = 0.0
    iters = 0
    first_state = session.run(model.init_first_state)
    second_state = session.run(model.init_second_state)
    fetches = {
            "cost": model.cost,
#            "x": model.x,
#            "y": model.y,
#            "steps": model.steps,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        '''
        for i, (c, h) in enumerate(model.init_first_state):
            feed_dict[c] = first_state[i].c
            feed_dict[h] = first_state[i].h

        for i, (c, h) in enumerate(model.init_second_state):
            feed_dict[c] = second_state[i].c
            feed_dict[h] = second_state[i].h
        #'''
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]

        costs += cost
        iters += 1
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f" %
                        (step * 1.0 / model.input.epoch_size, (costs / iters)))

    return (costs / iters)

def run_predict(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    fetches = {
            "x": model.x,
            "y": model.y,
            "probs": model.probs,
    }
    first_state = session.run(model.init_first_state)
    second_state = session.run(model.init_second_state)
    feed_dict = {}
    vals = session.run(fetches, feed_dict)
    x = vals["x"]
    y = np.reshape(vals["y"], (-1))
    _pos = np.where(y == 2)[0][0]
    y = y[0:_pos+1]
    probs = vals["probs"]
    return x, y.tolist(), probs

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 0
    hidden_size = 256
    max_epoch = 6
    max_max_epoch = 1000
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 50
    vocab_size = 3000
    frame_num = 80
    feat_size = 4096
    data_size = 0
    sent_len = 0

def main(_):
    config = MediumConfig()
    eval_config = MediumConfig()
    print "--- Read Train Data ---"
    train_raw_data, total_words, sent_len = read_file(config, True)
    print "--- Read Test Data ---"
    test_raw_data, _, _ = read_file(config, False)
    print "--- Build Words Dict ---"
    words_dic, rev_words_dic = build_words_dic(total_words, config)

    to_word = lambda ind: rev_words_dic[ind]
    # to_num = lambda word: words_dic.get(word, 0)
    i=0
    print "Train Data size ",len(train_raw_data)
    print "Test Data size ",len(test_raw_data)
    config.data_size = len(train_raw_data)
    config.sent_len = sent_len
    eval_config.data_size = 2
    eval_config.sent_len = sent_len
    eval_config.batch_size = 1
    #'''
    print "--- Data Sample ---"
    for key, _dic in test_raw_data.iteritems():
        i+=1
        if i>2:
            break
        print "Key: ",key
        print "Sent: _BOS %s" % (' '.join(_dic['sent'][0]))
        print "NP Shape: ",_dic['feat'].shape
        print "NP Type: ",_dic['feat'].dtype
        print _dic['feat'][0, 0:10]
    #'''
    train_data, train_valid = build_dataset(train_raw_data, \
                                                words_dic, config)
    test_data, test_valid = build_dataset(test_raw_data, \
                                                words_dic, config)
    #'''
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
            config.init_scale)

        with tf.name_scope("Train"):
            train_input = Input(config=config, data=train_data, \
                                    shuffle=True, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = S2VTModel(is_training=True, config=config, input_=train_input)

        with tf.name_scope("Test"):
            test_input = Input(config=eval_config, data=test_valid, \
                                    shuffle=False, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = S2VTModel(is_training=False, config=eval_config, input_=test_input)

        sv = tf.train.Supervisor(logdir=None)
        saver=sv.saver
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                fuck = 1
                #lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                #lr_decay = 0.002 * (0.97**i)
                #m.assign_lr(session, config.learning_rate * lr_decay)

                #print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, \
                                        eval_op=m.train_op, verbose=True)
                if ((i+1) % 20) == 0:
                    print("Epoch %d Train Perplexity: %.3f" % ((i+1), train_perplexity))
                if ((i+1) % 100) == 0:
                    print "==== Epoch %d ====" % (i+1)
                    for i in range(2):
                        x, y, probs = run_predict(session, mtest)
                        print "----------\nX Data: ",x[0, 0, 0:5]
                        print "[Ans] %s" % ( ' '.join( map(to_word, y) ) )
                        pred_words = np.argmax(probs, axis=1)
                        _pos = np.where(pred_words == 2)[0][0]
                        pred_list = pred_words[0:_pos+1].tolist()
                        print "[Pred] %s" % ( ' '.join( map(to_word, pred_list) ) )
    #'''
if __name__ == "__main__":
    tf.app.run()
