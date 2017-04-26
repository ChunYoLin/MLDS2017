import os
import re
import json
import collections
import numpy as np
import time
import math
import tensorflow as tf
import inspect


flags = tf.flags
logging = tf.logging

# Flags : TF's command line module, ==> (--model, default, help description)

flags.DEFINE_string("train_label", None,
                    "Train Label File.")
flags.DEFINE_string("feat_path", None,
                    "Feat directory.")
flags.DEFINE_string("id_file", None,
                    "Video ID file.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("train", False,
                  "Training model or Loading model")
flags.DEFINE_bool("sample", True,
                  "Use Sample Scheduling")
flags.DEFINE_string("out", None,
                  "Write out file")

FLAGS = flags.FLAGS

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

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reverse_dictionary
    # dictionary : store the word and its word's index in count
    # to_num = lambda word: dictionary.get(word, 0)
    # to_word = lambda ind: reverse_dictionary[ind]

def read_total_words(label):
    total_words = []
    with open(label, 'r') as F:
        label_data = json.load(F)
        for _dic in label_data:
            sent_list = _dic['caption']

            for sent in sent_list:
                words = parse_sent(sent)
                total_words.extend(words)

        del label_data
    return total_words

def read_file(config, id_list, path):
    data = {}
    total_words = []
    max_len = 0
    with open(id_list, 'r') as F:
        for _line in F.readlines():
            _id = _line[:-1]
            if _id in data:
                print "[Error] Label Data id Already in data"

            filename = _id + '.npy'
            _feat = np.load( path + filename )
            _x = {}
            _x['feat'] = _feat.astype( data_type(1) )
            data[_id] = _x

    return data

def build_dataset(data, words_dic, config, is_training):
    to_num = lambda word: words_dic.get(word, 0)

    sent_len = config.sent_len
    frame_num = config.frame_num
    feat_size = config.feat_size
    input_list = []
    target_list = []
    index_list = []
    i=0
    for key, _dic in data.iteritems():
        if is_training:
            _input = np.zeros((1, frame_num, feat_size), dtype=data_type(1))
            _input[0,:,:] = _dic['feat']
            input_list.append(_input)
            for _sent in _dic['sent']:
                _target = np.full((1, sent_len), 2, dtype=np.int)
                _list = map(to_num,  _sent)
                _target[0,0] = 1
                _target[0,1:len(_list)+1] = _list
                _index = np.array([i])
                target_list.append(_target)
                index_list.append(_index)
        else:
            _input = np.zeros((1, frame_num, feat_size), dtype=data_type(1))
            _target = np.full((1, sent_len), 2, dtype=np.int)
            _input[0,:,:] = _dic['feat']
            _target[0,0] = 1
            _index = np.array([i])
            input_list.append(_input)
            target_list.append(_target)
            index_list.append(_index)
        i+=1

    inputs = np.concatenate(input_list, axis=0)
    targets = np.concatenate(target_list, axis=0)
    index = np.concatenate(index_list, axis=0)
    return [inputs, targets, index], \
                    [inputs[0:3, :], targets[0:3, :], index[0:3]]

def data_producer(data, data_size, sent_len, \
                        frame_num, feat_size, batch_size, shuffle, name=None):
    with tf.name_scope(name, "DataProducer", \
            [data, data_size, sent_len, frame_num, feat_size, batch_size, shuffle]):

        x_data = data[0]
        y_data = data[1]
        index_data = data[2]

        i = tf.train.range_input_producer(data_size, shuffle=shuffle).dequeue()
        index_tensor = tf.convert_to_tensor(index_data, name="index_data", \
                                                dtype = tf.int32)
        index = tf.strided_slice(index_tensor, [i], [i+1])
        index = tf.reshape(index, [1])
        index.set_shape([1])
        x_tensor = tf.convert_to_tensor(x_data, name="x_data", \
                                                        dtype=data_type(0))
        x = tf.strided_slice(x_tensor, [index[0], 0, 0], [(index[0]+1), frame_num, feat_size])
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
        self.epoch_size = int( math.ceil(data_size / batch_size) )
        #self.epoch_size = int(data_size)
        self.input_data, self.targets = data_producer(
                data, data_size, sent_len, frame_num, feat_size, \
                                        batch_size, shuffle, name=name)
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
        self._sample_prob = tf.Variable(1.0, trainable=False)
        self._new_sample_prob = tf.placeholder( \
                        tf.float32, shape=[], name="new_sample_prob")
        self._sample_prob_update = tf.assign( \
                        self._sample_prob, self._new_sample_prob)
        # 1: For Frame to Layer 2 input
        weight_1 = tf.get_variable("weight_1", [feat_size, size], \
                                            dtype=data_type(0))
        # 2: For Layer 1 Frame output to Layer 2 and concat with text input
        weight_2 = tf.get_variable("weight_2", [size, size/2], \
                                            dtype=data_type(0))
        # A: For Global Location
        weight_a = tf.get_variable("weight_a", [size, frame_num], \
                                            dtype=data_type(0))
        # C: For concat info vector C and Layer 2 hidden state
        weight_c = tf.get_variable("weight_c", [size*2, size], \
                                            dtype=data_type(0))
        with tf.device("/cpu:0"):
            text_padding = tf.zeros([batch_size, frame_num, size/2], data_type(0))
            embedding = tf.get_variable("embedding", [vocab_size, size/2], \
                                            dtype=data_type(0))
            targets = tf.nn.embedding_lookup(embedding, input_.targets)
            text_inputs = tf.concat([text_padding, targets], axis = 1)

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type(0))
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type(0))

        def lstm_cell():
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

        inputs = tf.reshape(input_.input_data, (-1, feat_size))
        inputs = tf.matmul(inputs, weight_1)
        inputs = tf.reshape(inputs, (batch_size, frame_num, size))
        frame_inputs = tf.concat([inputs, frame_padding], axis = 1)
        if is_training and config.keep_prob < 1:
            frame_inputs = tf.nn.dropout(frame_inputs, config.keep_prob)
        #'''
        frame_info_list = []
        outputs = []
        last_index = []
        with tf.variable_scope("S2VT"):
            last_word_index = tf.zeros((batch_size), dtype=tf.int32)
            for time_step in range(num_steps-1):
                # First Layer RNN
                if time_step > 0 : tf.get_variable_scope().reuse_variables()
                (first_output, first_state) = \
                        first_cell(frame_inputs[:, time_step, :], first_state)
                second_input_1 = tf.matmul(first_output, weight_2)

                def target_input(): return text_inputs[:, time_step, :]
                def predict_input(): 
                    last_index.append(last_word_index)
                    return tf.nn.embedding_lookup(embedding, last_word_index)
                if time_step > frame_num:
                    if not is_training:
                        second_input_2 = predict_input()
                    elif FLAGS.sample:
                        _rand = tf.random_uniform([1])[0]
                        second_input_2 = tf.cond(_rand > self._sample_prob, \
                                    target_input, predict_input)
                    else:
                        second_input_2 = target_input()
                else:
                    second_input_2 = target_input()
                second_input = tf.concat([second_input_1, second_input_2], axis=1)

                # Second Layer RNN
                tf.get_variable_scope().reuse_variables()
                (second_output, second_state) = \
                                second_cell(second_input, second_state)
                # Only get Decoding stage Output.
                if time_step >= frame_num:
                    if time_step == frame_num:
                        _last_output = tf.reshape(second_output, [-1, size])
                        frame_info = tf.reshape(tf.concat(axis=1, values=frame_info_list), \
                                                                                [-1, frame_num, size])
                    _last_output = tf.reshape(second_output, [-1, size])
                    _hidden_t = tf.reshape(second_output, [-1, size])
                    # Hidden t [100, 256]
                    _location = tf.nn.softmax( tf.matmul(_hidden_t, weight_a) )
                    # Location [100, 80]
                    _vector_c_list = []
                    for _b in range(batch_size):
                        _weight = tf.reshape( _location[_b,:], [1, frame_num] )
                        _info = tf.reshape( frame_info[_b, :, :], [frame_num, size])
                        _c = tf.matmul( _weight, _info )
                        _vector_c_list.append( _c )
                    _vector_c = tf.reshape( tf.concat(axis=0, values=_vector_c_list), [-1, size])
                    # Vector C [100, 256]
                    _c_h = tf.concat(axis=1, values=[_vector_c, _hidden_t])
                    # _c_h [100, 512]
                    _output = tf.nn.tanh( tf.matmul( _c_h, weight_c ) )
                    # _output [100, 256]
                    _logits = tf.matmul( _output, softmax_w) + softmax_b
                    _probs = tf.nn.softmax( _logits )
                    last_word_index = tf.argmax( _probs, axis=1 )
                    outputs.append(_output)
                else:
                    frame_info_list.append(second_output)

        index = tf.reshape(tf.concat(axis=0, values=last_index), [-1])
        self._index = index
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        logits = tf.matmul(output, softmax_w) + softmax_b

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                                [logits],
                                [tf.reshape(input_.targets[:, 1:], [-1])],
                                [tf.ones([batch_size*(sent_len-1)], dtype=data_type(0))])

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._probs = tf.nn.softmax(logits)
        if not is_training:
            return
        self._train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

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
    def train_op(self):
        return self._train_op

    @property
    def sample_prob(self):
        return self._sample_prob

    @property
    def index(self):
        return self._index

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
            "index": model.index,
    }
    feed_dict = {}
    vals = session.run(fetches, feed_dict)
    x = vals["x"]
    y = np.reshape(vals["y"], (-1))
    try:
        _pos = np.where(y == 2)[0][0]
    except IndexError:
        _pos = 3
    y = y[0:_pos+1]
    probs = vals["probs"]
    #index = vals["index"]
    return x, y.tolist(), probs#, index

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
    sent_len = 42

def main(_):
    config = MediumConfig()
    eval_config = MediumConfig()
    if not FLAGS.train_label:
        print "[Error] Train Label File = none"
    total_words = read_total_words(FLAGS.train_label)

    if not FLAGS.id_file:
        print "[Error] ID_LIST File = none"
    if not FLAGS.feat_path:
        print "[Error] Feature Path = none"
    test_raw_data = read_file(config, FLAGS.id_file, FLAGS.feat_path)

    words_dic, rev_words_dic = build_words_dic(total_words, config)

    to_word = lambda ind: rev_words_dic[ind]
    # to_num = lambda word: words_dic.get(word, 0)
    test_data, test_valid = build_dataset(test_raw_data, \
                                                words_dic, config, False)
    test_data_size = len(test_data[2])
    eval_config.data_size = test_data_size
    eval_config.batch_size = 1

    #'''
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
            config.init_scale)
        with tf.name_scope("Test"):
            test_input = Input(config=eval_config, data=test_data, \
                                    shuffle=False, name="TestInput")
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                mtest = S2VTModel(is_training=False, config=eval_config, input_=test_input)

        sv = tf.train.Supervisor(logdir=None)
        saver=sv.saver
        start_time = time.time()
        with sv.managed_session() as session:
            if (not FLAGS.train) and FLAGS.save_path:
                saver.restore(session, FLAGS.save_path)
                ans_list = []
                print "===== Testing Data ====="
                for i in range(eval_config.data_size):
                    _ans_dict = {}
                    x, y, probs = run_predict(session, mtest)
                    pred_words = np.argmax(probs, axis=1)
                    _pos = np.where(pred_words == 2)[0][0]
                    pred_list = pred_words[0:_pos].tolist()
                    cand_sent = ' '.join( map(to_word, pred_list) )
                    _key = test_raw_data.keys()[i]
                    _ans_dict['caption'] = cand_sent
                    _ans_dict['id'] = _key
                    ans_list.append(_ans_dict)

                with open("output.json", "wb") as F:
                    json.dump(ans_list, F, separators=(',\n',': '))
    print "END"
    #'''
if __name__ == "__main__":
    tf.app.run()
