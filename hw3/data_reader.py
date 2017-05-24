import re
import csv
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np
import cPickle as pk
import random
from skip_thoughts import skipthoughts


class realimg(object):
    def __init__(self, img, match_sent):
        self.img = img
        self.wimg = []
        self.match_sent = match_sent
        self.mismatch_sent = []
        
    def sent2embed(self, model):
        match_sent = self.match_sent
        if match_sent:
            self.match_embed = skipthoughts.encode(model, match_sent)

        mismatch_sent = self.mismatch_sent
        if mismatch_sent:
            self.mismatch_embed = skipthoughts.encode(model, mismatch_sent)

def get_train_batch(img_objs, batch_size=64):
    data_size = len(img_objs)
    i = tf.train.range_input_producer(data_size, shuffle=False).dequeue()

    img = []
    wimg = []
    match_embed = []
    mismatch_embed = []
    for obj in img_objs:
        img.append(obj.img)
        wimg.append(obj.wimg[0])
        match_embed.append(obj.match_embed[0])
        mismatch_embed.append(obj.mismatch_embed[0])
    img = np.asarray(img)
    wimg = np.asarray(wimg)
    match_embed = np.asarray(match_embed)
    mismatch_embed = np.asarray(mismatch_embed)

    img_tensor = tf.convert_to_tensor(img, name='img_data', dtype=tf.float32)
    img_data = tf.strided_slice(img_tensor, [i, 0, 0, 0], [(i+1), 64, 64, 3])
    img_data = tf.reshape(img_data, [64, 64, 3])
    img_data.set_shape([64, 64, 3])

    wimg_tensor = tf.convert_to_tensor(wimg, name='wrong_img_data', dtype=tf.float32)
    wimg_data = tf.strided_slice(wimg_tensor, [i, 0, 0, 0], [(i+1), 64, 64, 3])
    wimg_data = tf.reshape(wimg_data, [64, 64, 3])
    wimg_data.set_shape([64, 64, 3])

    match_embed_tensor = tf.convert_to_tensor(
        match_embed, name='match_embed_data', dtype=tf.float32)
    match_embed_data = tf.strided_slice(match_embed_tensor, [i, 0], [(i+1), 4800])
    match_embed_data = tf.reshape(match_embed_data, [4800])
    match_embed_data.set_shape([4800])

    mismatch_embed_tensor = tf.convert_to_tensor(
        mismatch_embed, name='mismatch_embed_data', dtype=tf.float32)
    mismatch_embed_data = tf.strided_slice(mismatch_embed_tensor, [i, 0], [(i+1), 4800])
    mismatch_embed_data = tf.reshape(mismatch_embed_data, [4800])
    mismatch_embed_data.set_shape([4800])

    img_batch, wimg_batch, match_embed_batch, mismatch_embed_batch = tf.train.batch(
        [img_data, wimg_data, match_embed_data, mismatch_embed_data], batch_size=batch_size)

    return img_batch, wimg_batch, match_embed_batch, mismatch_embed_batch

def get_test_sent(test_file):
    with open(test_file, "r") as f:
        test_sent = []
        for row in f.read().splitlines():
            test_sent.append(row.split(",")[1])
        model = skipthoughts.load_model()
        vecs = skipthoughts.encode(model, test_sent)
        return vecs
    
def build_imgs():
    with open('./data/tags_clean.csv', 'r') as tag_file:
        tag_reader = csv.reader(tag_file, delimiter='\t')
        img_objs = []
        colors = [
            "red", "orange", "yellow", "green", "blue", "purple", "blonde",
            "pink", "black", "white", "brown"]
        num = 0
        print "generate captions from training tags....."
        for row in tag_reader:
            img_id = row[0].split(',')[0]
            tag_row = [row[0].split(',')[1]] + row[1:]
            img = skimage.io.imread(
                './data/faces/{}.jpg'.format(int(img_id)))
            img = skimage.transform.resize(img, (64, 64))
            match_sent = []
            mismatch_sent = []
            tag_hair = []
            tag_eyes = []
            for tag in tag_row:
                tag = tag.split(':')[0]
                for color in colors:
                    if "{} hair".format(color) in tag:
                        tag_hair.append(tag)
                    if "{} eyes".format(color) in tag:
                        tag_eyes.append(tag)
            for t_h in tag_hair:
                for t_e in tag_eyes:
                    r = random.random()
                    if r > 0.5:
                        match_sent.append('{} {}'.format(t_h, t_e))
                    else:
                        match_sent.append('{} {}'.format(t_e, t_h))
            if match_sent:
                #  print match_sent
                img_objs.append(realimg(img, match_sent))
                num += 1
                #  if num >= 64: break
                #  print match_sent
        model = skipthoughts.load_model()
        k = 0
        for idx, img_obj1 in enumerate(img_objs):
            find = 0
            for img_obj2 in img_objs[1:]:
                for sent in img_obj2.match_sent:
                    if sent not in img_obj1.match_sent:
                        img_objs[idx].wimg.append(img_obj2.img)
                        img_obj1.mismatch_sent.append(sent)
                        find += 1
                    if find >= 1: break
                if find >= 1: break
            img_obj1.sent2embed(model)
            print "{}/{}".format(k, len(img_objs))
            k += 1
    with open("./train_data/img_objs_new.pk", "w") as f:
        pk.dump(img_objs, f)


