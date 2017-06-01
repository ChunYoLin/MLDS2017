import re
import ast
import collections
import random

converations_path = './data/cornell movie-dialogs corpus/movie_conversations.txt'
lines_path = './data/cornell movie-dialogs corpus/movie_lines.txt'

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
        for line in movie_lines.read().splitlines():
            idx = line.split()[0]
            if len(line.split()) > 8:
                lines[idx] = line.split()[8:]
            else:
                lines[idx] = []
            speakers[idx] = line.split()[6]

    words = []
    for convs in conversations:
        for line_idx in convs:
            for word in lines[line_idx]:
                words.append(word)
    count = [['UNK', -1], ['BOS', -1], ['EOS', -1]]
    count.extend(collections.Counter(words).most_common(10000))
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
            line2id = [word_dict['BOS']]
            for word in lines[line_idx]:
                if word in word_dict:
                    line2id.append(word_dict[word])
                else:
                    line2id.append(word_dict['UNK'])
            lines2id.append(line2id)
        convs2id.append(lines2id)
    return convs2id, word_dict, inv_word_dict

convs2id, word_dict, inv_word_dict = read_raw()

def build_batch(convs2id, word_dict, batch_size=4, data_size=1280):
    convs_choosed = random.sample(range(len(convs2id)), 1000)
    encoder_input_batch = []
    decoder_input_batch = []
    decoder_target_batch = []
    encoder_input_batchs = []
    decoder_input_batchs = []
    decoder_target_batchs = []
    for idx, convs_id in enumerate(convs_choosed):
        encoder_input_batch.append(convs2id[convs_id][0])
        decoder_input_batch.append(convs2id[convs_id][1])
        decoder_target_batch.append(convs2id[convs_id][1][1:]+[word_dict['EOS']])
        if (idx+1) % batch_size == 0:
            max_len = 0
            for line in encoder_input_batch:
                if len(line) > max_len:
                    max_len = len(line)
            for line in encoder_input_batch:
                for _ in range(max_len-len(line)):
                   line.append(0) 

            max_len = 0
            for line in decoder_input_batch:
                if len(line) > max_len:
                    max_len = len(line)
            for line in decoder_input_batch:
                for _ in range(max_len-len(line)):
                   line.append(0) 

            max_len = 0
            for line in decoder_target_batch:
                if len(line) > max_len:
                    max_len = len(line)
            for line in decoder_target_batch:
                for _ in range(max_len-len(line)):
                   line.append(0) 
                    
            encoder_input_batchs.append(encoder_input_batch)
            decoder_input_batchs.append(decoder_input_batch)
            decoder_target_batchs.append(decoder_target_batch)
            encoder_input_batch = []
            decoder_input_batch = []
            decoder_target_batch = []

    return encoder_input_batchs, decoder_input_batchs, decoder_target_batchs
encoder_input_batchs , decoder_input_batchs , decoder_target_batchs = (
    build_batch(convs2id, word_dict))
print len(decoder_target_batchs[0])
print len(decoder_input_batchs[0])
            

