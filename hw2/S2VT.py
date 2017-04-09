import os
import re
import json
import collections
import numpy as np
import time
import tensorflow as tf
import inspect

PATH = "Data/training_data/feat/"
LABEL = "Data/training_label.json"

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
    count.extend(c.most_common(config.vocab_size - 1))
    print "Raw Vocab Size %d" % len(c)
    print "Total Vocab Size %d" % len(count)

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reverse_dictionary
    # Data : Like actual word list, but store the word's index in count
    # Count : store the word and its occur numbers. from biggest to smallest
    # dictionary : store the word and its word's index in count
    # to_num = lambda word: dictionary.get(word, 0)
    # to_word = lambda ind: reverse_dictionary[ind]

def read_file(config):
    data = {}
    total_words = []
    max_len = 0
    with open(LABEL, 'r') as F:
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
            _feat = np.load(PATH+filename)
            _x['feat'] = _feat.astype( data_type(1) )
            data[_dic['id']] = _x

        del label_data
    print "--- Read Label ---"
    print "Data Len: ",len(data)
    print "Total Words num: ",len(total_words)
    print "Max sent len: ",max_len
    print "--- Build Words Dict ---"
    words_dic, rev_words_dic = build_words_dic(total_words, config)
    return data, words_dic, rev_words_dic, max_len+2

def build_dataset(data, words_dic, config):
    to_num = lambda word: words_dic.get(word, 0)

    data_size = config.data_size
    sent_len = config.sent_len
    _input = np.zeros((data_size, 80, 4096), dtype=data_type(1))
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
    return [_input, _target, _step_list]

def data_producer(raw_data, data_size, sent_len, \
                        frame_num, feat_size, name=None):
    with tf.name_scope(name, "DataProducer", [raw_data, data_size, sent_len]):
        raw_x_data = raw_data[0]
        raw_y_data = raw_data[1]
        raw_steps_data = raw_data[2]

        x_data = tf.convert_to_tensor(raw_x_data, name="raw_x_data", \
                                                        dtype=data_type(0))
        i = tf.train.range_input_producer(data_size, shuffle=False).dequeue()
        x = tf.strided_slice(x_data, [i, 0, 0], [(i+1), frame_num, feat_size])
        x = tf.reshape(x, [frame_num, feat_size])
        x.set_shape([frame_num, feat_size])
        y_data = tf.convert_to_tensor(raw_y_data, name="raw_y_data", \
                                                        dtype=tf.int32)
        y = tf.strided_slice(y_data, [i, 0], [(i+1), sent_len])
        y = tf.reshape(y, [sent_len])
        y.set_shape([sent_len])

        steps_data = tf.convert_to_tensor(raw_steps_data, name="raw_steps_data", \
                                                        dtype=tf.int32)
        steps = tf.gather(steps_data, [i])
        steps = tf.to_int32(steps)
        return x, y, steps

class Input(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.frame_num = frame_num = config.frame_num
        self.feat_size = feat_size = config.feat_size
        self.data_size = data_size = config.data_size
        self.sent_len = sent_len = config.sent_len
        self.num_steps = frame_num + sent_len
        self.epoch_size = data_size
        self.input_data, self.targets, self.text_steps = data_producer(
                data, data_size, sent_len, frame_num, feat_size, name=name)
        # Input_data = video frame feature,  Target = Video decription ( 1 sentence )
        # With producer, each run epoch will produce one data 
        # Input_data = [80, 4096], Target = [Sent_len]

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
        self._steps = steps = input_.text_steps

        frame_padding = tf.zeros([sent_len, size], tf.float32)
        text_padding = tf.zeros([frame_num, size/2], data_type(0))

        weight_1 = tf.get_variable("weight_1", [feat_size, size], \
                                            dtype=data_type(0))
        weight_2 = tf.get_variable("weight_2", [size, size/2], \
                                            dtype=data_type(0))
        embedding = tf.get_variable("embedding", [vocab_size, size/2], \
                                            dtype=data_type(0))
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


        first_cell = lstm_cell()
        second_cell = lstm_cell()
        self._init_first_state = first_cell.zero_state(1, data_type(0))
        self._init_second_state = second_cell.zero_state(1, data_type(0))
        first_state = self._init_first_state
        second_state = self._init_second_state

        targets = tf.nn.embedding_lookup(embedding, input_.targets)
        inputs = tf.matmul(input_.input_data, weight_1)
        frame_inputs = tf.concat([inputs, frame_padding], axis = 0)
        text_inputs = tf.concat([text_padding, targets], axis = 0)
        #'''
        outputs = []
        with tf.variable_scope("S2VT"):
            for time_step in range(num_steps):
                # First Layer RNN
                if time_step > 0 : tf.get_variable_scope().reuse_variables()
                first_input = tf.reshape(frame_inputs[time_step, :], [-1, size])
                (first_output, first_state) = \
                                first_cell(first_input, first_state)
                first_output = tf.reshape(first_output, [1, size])
                second_input_1 = tf.matmul(first_output, weight_2)
                second_input_2 = tf.reshape(text_inputs[time_step, :], [1, size/2])
                second_input = tf.concat([second_input_1, second_input_2], axis=1)

                # Second Layer RNN
                tf.get_variable_scope().reuse_variables()
                (second_output, second_state) = \
                                second_cell(second_input, second_state)
                # Only get Decoding stage Output.
                if time_step >= frame_num:
                    outputs.append(second_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        print output.shape
        print steps.shape
        output = tf.reshape(output[0:steps[0]-1, :], [-1, size])
        print output.shape
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type(0))
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type(0))

        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                                [logits],
                                [tf.reshape(input_.targets[1:steps[0]], [-1])],
                                [tf.ones([steps[0]-1], dtype=data_type(0))])
        self._cost = cost = tf.reduce_sum(loss)
        self._probs = tf.nn.softmax(logits)
        #'''
        if not is_training:
            return
        #'''
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                    config.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())
        #optimizer = tf.train.AdamOptimizer(0.001)
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

#    @property
#    def train_op(self):
#        return self._train_op

def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    costs = 0.0
    iters = 0
    fetches = {
            "cost": model.cost,
#            "x": model.x,
#            "y": model.y,
#            "steps": model.steps,
    }
    #if eval_op is not None:
    #    fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        first_state = session.run(model.init_first_state)
        second_state = session.run(model.init_second_state)
        '''
        for i, (c, h) in enumerate(model.init_first_state):
            feed_dict[c] = first_state[i].c
            feed_dict[h] = first_state[i].h

        for i, (c, h) in enumerate(model.init_second_state):
            feed_dict[c] = second_state[i].c
            feed_dict[h] = second_state[i].h
        '''
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
            "steps": model.steps,
            "probs": model.probs,
    }
    first_state = session.run(model.init_first_state)
    second_state = session.run(model.init_second_state)
    feed_dict = {}
    vals = session.run(fetches, feed_dict)
    x = vals["x"]
    y = np.reshape(vals["y"], (-1))
    steps = vals["steps"]
    probs = vals["probs"]
    return x, y[0:steps].tolist(), steps, probs

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 0
    hidden_size = 256
    max_epoch = 6
    max_max_epoch = 10
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 6
    vocab_size = 3000
    frame_num = 80
    feat_size = 4096
    data_size = 0
    sent_len = 0

if __name__ == "__main__":
    config = MediumConfig()
    data, words_dic, rev_words_dic, sent_len = read_file(config)
    to_word = lambda ind: rev_words_dic[ind]
    # to_num = lambda word: words_dic.get(word, 0)
    i=0
    print "Data size ",len(data)
    config.data_size = len(data)
    config.sent_len = sent_len
    #'''
    for key, _dic in data.iteritems():
        i+=1
        if i>2:
            break
        print "Key ",key
        print "Sent : "
        print ' '.join(_dic['sent'][0])
        print "NP Shape : ",_dic['feat'].shape
        print "NP Type : ",_dic['feat'].dtype
        print _dic['feat'][0, 0:10]
    #'''
    train_data = build_dataset(data, words_dic, config)
    #'''
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
            config.init_scale)

        with tf.name_scope("Train"):
            train_input = Input(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = S2VTModel(is_training=True, config=config, input_=train_input)

        sv = tf.train.Supervisor(None)
        saver=sv.saver
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                #'''
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                #lr_decay = 0.002 * (0.97**i)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, verbose=True)
                print "==== Epoch %d ====" % (i+1)
                print("Train Perplexity: %.3f" % (train_perplexity))
                #'''
                #'''
                x, y, steps, probs = run_predict(session, m)
                print "X shape: ",x.shape
                print "X Data: ",x[0, 0:5]
                print ' '.join( map(to_word, y) )
                print "Steps : %d" % steps
                pred_word = np.argmax(probs, axis=1)
                print ' '.join( map(to_word, pred_word) )
                #'''
