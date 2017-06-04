import re
import ast
import collections
import random

import numpy as np

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
lemmatiser = WordNetLemmatizer()

import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

converations_path = './data/cornell movie-dialogs corpus/movie_conversations.txt'
lines_path = './data/cornell movie-dialogs corpus/movie_lines.txt'
selected_path = './data/movie_lines_selected.txt'
_bucket = [(5, 10), (10, 15), (20, 25), (40, 50)]

def build_word_dict(words, vocab_size):
    count = [['UNK', -1], ['GO', -1], ['EOS', -1], ['PAD', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size-4))
    word_dict = {}
    inv_word_dict = {}
    for word, _ in count:
        idx = len(word_dict)
        word_dict[word] = idx
        inv_word_dict[idx] = word
    return word_dict, inv_word_dict

def read_selected(data_size):
    data_set = [[] for _ in _bucket]
    with open(selected_path, 'r') as movie_lines:
        all_lines = movie_lines.readlines()
        source_raw = all_lines[0::2]
        target_raw = all_lines[1::2]
        counter = 0
        words = []
        #  build word dict
        for idx in range(len(source_raw)):
            source_raw[idx] = nltk.word_tokenize(source_raw[idx].lower())
            for word in source_raw[idx]:
                words.append(word)
        for idx in range(len(target_raw)):
            target_raw[idx] = nltk.word_tokenize(target_raw[idx].lower())
            for word in target_raw[idx]:
                words.append(word)
        word_dict, inv_word_dict = build_word_dict(words, 20000)
        #  convert source raw to id
        source = []
        for line in source_raw:
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
        return word_dict, inv_word_dict, data_set

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
                     for batch_idx in xrange(batch_size)], dtype=np.int32))
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == word_dict['PAD']:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

#  get_batch(w_id, a, 2, 4)

