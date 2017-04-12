import sys
import os
import re
import json
import collections
import tensorflow as tf
import numpy as np

def _build_vocab():
    with open("./MLDS_hw2_data/training_label.json") as train:
        train_json = json.load(train)
        words = []
        for i in range(len(train_json)):
            for caption in train_json[i]["caption"]:
                new_caption = 'BOS ' + caption[:len(caption) - 1] + ' EOS'
                for word in new_caption.split():
                    words.append(word)
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(1000000))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        #  print dictionary
    return dictionary

def _target_data():
    feat_path = './MLDS_hw2_data/training_data/feat/'
    with open("./MLDS_hw2_data/training_label.json") as train:
        train_json = json.load(train)
        train_captions = []
        frame_data = {}
        for i in range(len(train_json)):
            caption = train_json[i]["caption"][0]
            train_caption = 'BOS ' + caption[:len(caption) - 1] + ' EOS'
            train_captions.append(train_caption)
            video_id = train_json[i]["id"] 
            feat_file_name = video_id + '.npy'
            feat = np.load(feat_path + feat_file_name)
            frame_data[video_id] = feat
    return frame_data, train_captions
_target_data()

def _target_data_to_word_id(corpus, word_to_id):
    max_len = 0
    caption_id = []
    for caption in corpus:
        new_caption = []
        caption = caption.split()
        for word in caption:
            new_caption.append(word_to_id[word])
        caption_id.append(new_caption)
        if len(caption) > max_len:
            max_len = len(caption)
            
    train_data = []
    for caption in caption_id:
        pad_caption = caption[:]
        for i in range(max_len - len(caption)):
            pad_caption.append(0)
        for w in pad_caption:
            train_data.append(w)
    train_data.append(0)
    return train_data, max_len
        

def Data_producer(frame_data, text_data, batch_size, num_steps, name = None):
    with tf.name_scope(name, "Data_producer", [frame_data, text_data, batch_size, num_steps]):
        text_data = tf.convert_to_tensor(text_data, name = "text_data", dtype = tf.int32)
        data_len = tf.size(text_data)
        batch_len = data_len // batch_size
        data = tf.reshape(text_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        i = tf.train.range_input_producer(epoch_size, shuffle = False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])

        return x, y
