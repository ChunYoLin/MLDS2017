import re
import ast
import collections
import random
import json
import pickle as pk

import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
lemmatiser = WordNetLemmatizer()

import sys
import os
reload(sys)
sys.setdefaultencoding("ISO-8859-1")
selected_path = '../Data/train_all.txt'

_bucket = [(5, 10), (10, 15), (20, 25), (40, 50)]

def build_w():
    words = []
    vocab_size = 50000
    with open(selected_path) as f:
        for line in f.read().splitlines():
            s = nltk.word_tokenize(line.lower())
            for w in s:
                words.append(w)
    count = [['UNK', -1], ['GO', -1], ['EOS', -1], ['PAD', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size-4))
    word_dict = {}
    inv_word_dict = {}
    for word, _ in count:
        idx = len(word_dict)
        word_dict[word] = idx
        inv_word_dict[idx] = word
    with open('./DICT/w_id.pk', 'w') as w, open('./DICT/inv_w_id.pk', 'w') as inv_w:
        pk.dump(word_dict, w)
        pk.dump(inv_word_dict, inv_w)

#  def build_word_dict(words, vocab_size):

    #  count = [['UNK', -1], ['GO', -1], ['EOS', -1], ['PAD', -1]]
    #  count.extend(collections.Counter(words).most_common(vocab_size-4))
    #  word_dict = {}
    #  inv_word_dict = {}
    #  for word, _ in count:
        #  idx = len(word_dict)
        #  word_dict[word] = idx
        #  inv_word_dict[idx] = word
    #  return word_dict, inv_word_dict

def read_lines(word_dict, path, data_size):
    data_set = [[] for _ in _bucket]
    with open(path, 'r') as movie_lines:
        all_lines = movie_lines.readlines()
        source_raw = all_lines[0::2]
        target_raw = all_lines[1::2]
        counter = 0
        words = []
        #  convert source raw to id
        source = []
        for line in source_raw:
            line = nltk.word_tokenize(line.lower())
            single_line = []
            for word in line:
                if word in word_dict:
                    single_line.append(word_dict[word])
                else:
                    single_line.append(0)
            source.append(single_line)
        #  convert target raw to id
        target = []
        for line in target_raw:
            line = nltk.word_tokenize(line.lower())
            single_line = []
            for word in line:
                if word in word_dict:
                    single_line.append(word_dict[word])
                else:
                    single_line.append(0)
            target.append(single_line)
        counter = 0
        line_idx = 0
        while source and target and (not counter > data_size):
            counter += 1
            if counter % 1000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source[line_idx]]
            target_ids = [int(x) for x in target[line_idx]]
            line_idx += 1
            target_ids.append(word_dict['EOS'])
            for bucket_id, (source_size, target_size) in enumerate(_bucket):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
        return data_set


def read_path(word_dict, path, data_size, file_num):
    data_set = [[] for _ in _bucket]
    source = []
    target = []
    i=0
    for filename in os.listdir(path):
        i+=1
        if i > file_num:
            break
        with open(path+filename, 'r') as movie_lines:
            all_lines = movie_lines.readlines()
            source_raw = all_lines[0::2]
            target_raw = all_lines[1::2]
            if len(target_raw) < len(source_raw):
                source_raw = source_raw[0:len(target_raw)]
            if len(source_raw) != len(target_raw):
                print "[Error] Source len < Target len"
            #  convert source raw to id
            for line in source_raw:
                line = nltk.word_tokenize(line.lower())
                single_line = []
                for word in line:
                    if word in word_dict:
                        single_line.append(word_dict[word])
                    else:
                        single_line.append(0)
                source.append(single_line)
            #  convert target raw to id
            for line in target_raw:
                line = nltk.word_tokenize(line.lower())
                single_line = []
                for word in line:
                    if word in word_dict:
                        single_line.append(word_dict[word])
                    else:
                        single_line.append(0)
                target.append(single_line)
    counter = 0
    line_idx = 0
    while (line_idx < len(source)) and (line_idx < len(target)) \
                                and (not counter > data_size):
        counter += 1
        if counter % 10000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
        source_ids = [int(x) for x in source[line_idx]]
        target_ids = [int(x) for x in target[line_idx]]
        line_idx += 1
        target_ids.append(word_dict['EOS'])
        for bucket_id, (source_size, target_size) in enumerate(_bucket):
            if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids])
                break
    return data_set

#  w_id, inv_w_id, a = read_selected(2000)
def get_batch(word_dict, data, bucket_id, batch_size):
    encoder_size, decoder_size = _bucket[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    for _ in xrange(batch_size):
        encoder_input, decoder_input = random.choice(data[bucket_id])
        encoder_pad = [word_dict['PAD']] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([word_dict['GO']] + decoder_input +
                              [word_dict['PAD']] * decoder_pad_size)
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                     for batch_idx in xrange(batch_size)], dtype=np.int32))
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                     for batch_idx in xrange(
                         batch_size)], dtype=np.int32).reshape(batch_size))
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == word_dict['PAD']:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights




