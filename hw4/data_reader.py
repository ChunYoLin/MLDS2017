import re
import ast
import collections
import random

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
lemmatiser = WordNetLemmatizer()

import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

converations_path = './data/cornell movie-dialogs corpus/movie_conversations.txt'
lines_path = './data/cornell movie-dialogs corpus/movie_lines.txt'
selected_path = './data/movie_lines_selected_10k.txt'

def read_selected():
    with open(selected_path, 'r') as movie_lines:
        all_lines = movie_lines.read().splitlines()
        encoder_input_raw = []
        decoder_input_raw = []
        for i in range(0, len(all_lines), 2):
            encoder_input = nltk.word_tokenize(all_lines[i].lower())
            decoder_input = nltk.word_tokenize(all_lines[i+1].lower())
            encoder_input_raw.append(encoder_input)
            decoder_input_raw.append(decoder_input)
        words = []
        for line in encoder_input_raw:
            for word in line:
                words.append(word)
        for line in decoder_input_raw:
            for word in line:
                words.append(word)
        count = [['UNK', -1], ['BOS', -1], ['EOS', -1]]
        count.extend(collections.Counter(words).most_common(10000))
        word_dict = {}
        inv_word_dict = {}
        for word, _ in count:
            idx = len(word_dict)
            word_dict[word] = idx
            inv_word_dict[idx] = word
        
        encoder_input_ids = []
        for line in encoder_input_raw:
            encoder_input_id = []
            for word in line:
                if word in word_dict:
                    encoder_input_id.append(word_dict[word])
                else:
                    encoder_input_id.append(0)
            encoder_input_ids.append(encoder_input_id)

        decoder_input_ids = []
        decoder_target_ids = []
        for line in decoder_input_raw:
            decoder_input_id = []
            decoder_target_id = []
            for idx, word in enumerate(line):
                if word in word_dict:
                    decoder_input_id.append(word_dict[word])
                    decoder_target_id.append(word_dict[word])
                else:
                    decoder_input_id.append(0)
                    decoder_target_id.append(0)
            decoder_input_ids.append(decoder_input_id)
            decoder_target_id.append(2)
            decoder_target_ids.append(decoder_target_id)

        return encoder_input_ids, decoder_input_ids, decoder_target_ids, word_dict, inv_word_dict

def build_selected_batch(
    encoder_input_ids, decoder_input_ids, decoder_target_ids, word_dict,
    batch_size=4):
    encoder_input_batch = []
    decoder_input_batch = []
    decoder_target_batch = []
    encoder_input_batchs = []
    decoder_input_batchs = []
    decoder_target_batchs = []
    for i in range(len(encoder_input_ids)):
        encoder_input_batch.append(encoder_input_ids[i])
        decoder_input_batch.append(decoder_input_ids[i])
        decoder_target_batch.append(decoder_target_ids[i])
        if (i+1) % batch_size == 0:
            max_len = 0
            for line in encoder_input_batch:
                if len(line) > max_len:
                    max_len = len(line)
            for line in encoder_input_batch:
                l = len(line)
                for _ in range(max_len-l):
                   line.append(0) 

            for line in decoder_input_batch:
                l = len(line)
                for _ in range(280-l):
                   line.append(0) 

            for line in decoder_target_batch:
                l = len(line)
                for _ in range(280-l):
                   line.append(0) 
                    
            encoder_input_batchs.append(encoder_input_batch)
            decoder_input_batchs.append(decoder_input_batch)
            decoder_target_batchs.append(decoder_target_batch)
            encoder_input_batch = []
            decoder_input_batch = []
            decoder_target_batch = []

    return encoder_input_batchs, decoder_input_batchs, decoder_target_batchs


def read_raw():
    conversations = []
    with open(converations_path, 'r') as meta:
        for line in meta.read().splitlines():
            s = ''
            for i in range(6, len(line.split()), 1):
                s += line.split()[i]
            convs = ast.literal_eval(s)
            conversations.append(convs)

    lines = {}
    speakers = {}
    with open(lines_path, 'r') as movie_lines:
        k = 0
        for line in movie_lines.read().splitlines():
            line = line.split(' +++$+++ ')
            idx = line[0]
            if len(line) > 4:
                s = line[4]
            else:
                s = ""
            s = s.lower()
            lines[idx] = nltk.word_tokenize(s)
            #  for w_id, w in enumerate(lines[idx]):
                #  lines[idx][w_id] = lemmatiser.lemmatize(w)
            #  lines[idx] = re.sub(',', '', lines[idx])
            #  lines[idx] = re.sub('\.', '', lines[idx])
            #  lines[idx] = re.sub('!', '', lines[idx])
            #  lines[idx] = re.sub('\?', '', lines[idx])
            speakers[idx] = line[3]
            k+=1
    words = []
    for convs in conversations:
        for line_idx in convs:
            for word in lines[line_idx]:
                words.append(word)
    count = [['UNK', -1], ['BOS', -1], ['EOS', -1]]
    count.extend(collections.Counter(words).most_common(50000))
    word_dict = {}
    inv_word_dict = {}
    for word, _ in count:
        idx = len(word_dict)
        word_dict[word] = idx
        inv_word_dict[idx] = word
    
    convs2id = []
    for convs in conversations:
        lines2id = []
        for line_idx in convs:
            line2id = []
            for word in lines[line_idx]:
                if word in word_dict:
                    line2id.append(word_dict[word])
                else:
                    line2id.append(word_dict['UNK'])
            lines2id.append(line2id)
        convs2id.append(lines2id)
    return convs2id, word_dict, inv_word_dict

def build_batch(convs2id, word_dict, batch_size=4, data_size=1280):
    #  convs_choosed = random.sample(range(len(convs2id)), 1000)
    encoder_input_batch = []
    decoder_input_batch = []
    decoder_target_batch = []
    encoder_input_batchs = []
    decoder_input_batchs = []
    decoder_target_batchs = []
    #  for idx, convs_id in enumerate(convs_choosed):
    convs = 0
    convs_choosed = set()
    while convs < data_size:
        convs_id = random.randint(0, len(convs2id)-1)
        target_len = len(convs2id[convs_id][1])
        if target_len > 5 and target_len < 100 and convs_id not in convs_choosed:
            convs_choosed.add(convs_id)
            convs += 1
            encoder_input_batch.append(convs2id[convs_id][0])
            decoder_input_batch.append(convs2id[convs_id][1])
            decoder_target_batch.append(convs2id[convs_id][1]+[word_dict['EOS']])
            if (convs) % batch_size == 0:
                max_len = 0
                for line in encoder_input_batch:
                    if len(line) > max_len:
                        max_len = len(line)
                for line in encoder_input_batch:
                    l = len(line)
                    for _ in range(max_len-l):
                       line.append(0) 

                for line in decoder_input_batch:
                    l = len(line)
                    for _ in range(100-l):
                       line.append(0) 

                for line in decoder_target_batch:
                    l = len(line)
                    for _ in range(100-l):
                       line.append(0) 
                        
                encoder_input_batchs.append(encoder_input_batch)
                decoder_input_batchs.append(decoder_input_batch)
                decoder_target_batchs.append(decoder_target_batch)
                encoder_input_batch = []
                decoder_input_batch = []
                decoder_target_batch = []

    return encoder_input_batchs, decoder_input_batchs, decoder_target_batchs
#  a, b, c = read_raw()
#  build_batch(a, b)

