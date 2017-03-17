import collections
import os
import re
import tensorflow as tf

f = './data/Holmes_Training_Data/14WOZ10.TXT'
def _read_words(filename):
    words = []
    with open(filename, 'r') as f:
        for word in f.read().replace("\n", "").lower().split():
            word = re.sub('!', '', word)
            word = re.sub('\?', '', word)
            word = re.sub('\.', '', word)
            word = re.sub('"', '', word)
            word = re.sub(';', '', word)
            words.append(word)
        return words

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key = lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def Data_producer(raw_data, batch_size, num_steps, name = None):
    with tf.name_scope(name, "Data_producer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name = "raw_data", dtype = tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        i = tf.train.range_input_producer(epoch_size, shuffle = False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
with tf.device('/cpu:0'):
    word_to_id = _build_vocab(f)
    print len(word_to_id)
    train = _file_to_word_ids(f, word_to_id)
    x, y = Data_producer(train, 32, 5)

    sess = tf.Session()
    sess.run(x)
