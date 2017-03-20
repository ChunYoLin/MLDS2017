import collections
import os
import re
import csv
import tensorflow as tf

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

def _read_multi_words(filelist):
    words = []
    for filename in filelist:
        with open(filename, 'r') as f:
            for word in f.read().replace("\n", "").lower().split():
                word = re.sub('!', '', word)
                word = re.sub('\?', '', word)
                word = re.sub('\.', '', word)
                word = re.sub('"', '', word)
                word = re.sub(';', '', word)
                words.append(word)
    return words

def _build_vocab(filename, vocab_size):
    data = _read_words(filename)
    count = [['UNK', -1]]
    count.extend(collections.Counter(data).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary

def _build_multi_vocab(filelist, vocab_size):
    data = _read_multi_words(filelist)
    count = [['UNK', -1]]
    count.extend(collections.Counter(data).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary

def _list_to_word_ids(_list, word_to_id):
    return [word_to_id[word] for word in _list if word in word_to_id]


def _file_to_word_ids(filename, word_to_id):
    words = _read_words(filename)
    data = list()
    unk_count = 0
    for word in words:
        if word in word_to_id:
            index = word_to_id[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    #  count[0][1] = unk_count
    return data
    #  return [word_to_id[word] for word in data if word in word_to_id]

def _multi_file_to_word_ids(filelist, word_to_id):
    words = _read_multi_words(filelist)
    data = list()
    unk_count = 0
    for word in words:
        if word in word_to_id:
            index = word_to_id[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    #  count[0][1] = unk_count
    return data
    #  return [word_to_id[word] for word in data if word in word_to_id]

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
