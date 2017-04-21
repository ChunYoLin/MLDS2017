import sys
import os
import re
import json
import collections
import tensorflow as tf
import numpy as np


def _read_train_data():
    feat_path = './MLDS_hw2_data/training_data/feat/'
    with open("./MLDS_hw2_data/training_label.json") as train:
        train_json = json.load(train)
        train_captions = []
        frame_data = []
        for i in range(len(train_json)):
            caption = train_json[i]["caption"][0]
            train_caption = caption[:len(caption) - 1] + ' EOS'
            train_caption = re.sub(",", " ,", train_caption)
            train_captions.append(train_caption)
            video_id = train_json[i]["id"] 
            feat_file_name = video_id + '.npy'
            feat = np.load(feat_path + feat_file_name)
            frame_data.append(feat)
        frame_data = np.asarray(frame_data, dtype = np.float32)
    return frame_data, train_captions

def _read_test_data():
    feat_path = './MLDS_hw2_data/testing_data/feat/'
    with open("./MLDS_hw2_data/testing_public_label.json") as test:
        test_json = json.load(test)
        test_captions = []
        frame_data = []
        for i in range(len(test_json)):
            caption = test_json[i]["caption"][0]
            test_caption = caption[:len(caption) - 1] + ' EOS'
            test_caption = re.sub(",", " ,", test_caption)
            test_captions.append(test_caption)
            video_id = test_json[i]["id"] 
            feat_file_name = video_id + '.npy'
            feat = np.load(feat_path + feat_file_name)
            frame_data.append(feat)
        frame_data = np.asarray(frame_data, dtype = np.float32)
    return frame_data, test_captions

def _read_time_limited_data():
    feat_path = './MLDS_hw2_time_limited/feat/'
    frame_data = []
    feat_files = []
    for feat_file_name in os.listdir(feat_path):
        feat_files.append(feat_file_name)
        feat = np.load(feat_path + feat_file_name)
        frame_data.append(feat)
    frame_data = np.asarray(frame_data, dtype = np.float32)
    return frame_data, feat_files

def _build_word_id():
    with open("./MLDS_hw2_data/training_label.json") as train:
        train_json = json.load(train)
        words = []
        for i in range(len(train_json)):
            for caption in train_json[i]["caption"]:
                new_caption = caption[:len(caption) - 1] + ' EOS'
                for word in new_caption.split():
                    words.append(word)
        count = [['UNK', -1], ['BOS', -1]]
        count.extend(collections.Counter(words).most_common(10000))
        dictionary = dict()
        inv_dictionary = dict()
        for word, _ in count:
            idx = len(dictionary)
            dictionary[word] = idx
            inv_dictionary[idx] = word
        del train_json
    return dictionary, inv_dictionary

def _text_data_to_word_id(text_data_raw, word_to_id):
    max_len = 0
    caption_id = []
    orig_sent_len = []
    for caption in text_data_raw:
        new_caption = []
        caption = caption.split()
        for word in caption:
            if word in word_to_id:
                new_caption.append(word_to_id[word])
            else:
                new_caption.append(0)
        caption_id.append(new_caption)
        orig_sent_len.append(len(caption))
        if len(caption) > max_len:
            max_len = len(caption)
    text_data = []
    for caption in caption_id:
        pad_caption = caption[:]
        for i in range(max_len - len(caption)):
            pad_caption.append(0)
        text_data.append(pad_caption)
    del caption_id
    return text_data, max_len, orig_sent_len

def Data_producer(frame_data, text_data, batch_size, sent_len, orig_sent_len, name = None):
    with tf.name_scope(name, "Data_producer", [frame_data, text_data, batch_size, sent_len]):
        data_size = len(frame_data)
        i = tf.train.range_input_producer(data_size, shuffle = False).dequeue()

        frame_feat_tensor = tf.convert_to_tensor(frame_data, name = "frame_feat_data", dtype = tf.float32)
        frame_feat_data = tf.strided_slice(frame_feat_tensor, [i, 0, 0], [(i + 1), 80, 4096])
        frame_feat_data = tf.reshape(frame_feat_data, [80, 4096])
        frame_feat_data.set_shape([80, 4096])

        text_id_tensor = tf.convert_to_tensor(text_data, name = "text_id_data", dtype = tf.int32)
        text_id_data = tf.strided_slice(text_id_tensor, [i, 0], [(i + 1), sent_len])
        text_id_data = tf.reshape(text_id_data, [sent_len])
        text_id_data.set_shape([sent_len])

        orig_sent_len_tensor = tf.convert_to_tensor(orig_sent_len, name = "orig_sent_len", dtype = tf.int32)
        orig_sent_len_data = tf.strided_slice(orig_sent_len, [i], [(i + 1)])
        orig_sent_len_data = tf.reshape(orig_sent_len_data, [1])
        orig_sent_len_data.set_shape([1])

        frame_batch, text_batch = tf.train.batch([frame_feat_data, text_id_data], batch_size = batch_size)
        return frame_batch, text_batch, orig_sent_len_data


