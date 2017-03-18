import collections
import os
import re
import csv
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
    count = [['UNK', -1]]
    count.extend(collections.Counter(data).most_common(11999))
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

def test_data():
    test_file = './data/testing_data.csv'
    with open(test_file)as f:
        test_reader = csv.reader(f, delimiter = ',')
        test_question_all = []
        test_question_before = []
        test_question_after = []
        test_answer = []
        for line in test_reader:
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
                line_before.append('UNK')
                line_after.append('UNK')
                test_question_before.append(line_before)
                test_question_after.append(line_after)
                test_question_all.append(line[1])
                test_answer.append(line[2:])
    return test_question_all, test_question_before, test_question_after, test_answer

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
