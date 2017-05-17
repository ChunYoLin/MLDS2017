import re
import csv
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import cPickle as pk
from skip_thoughts import skipthoughts


class realimg(object):
    def __init__(self, img, match_sent):
        self.img = img
        self.match_sent = match_sent
        self.mismatch_sent = []

    def sent2embed(self, model):
        match_sent = self.match_sent
        if match_sent:
            self.match_embed = skipthoughts.encode(model, match_sent)
        mismatch_sent = self.mismatch_sent
        if mismatch_sent:
            self.mismatch_embed = skipthoughts.encode(model, mismatch_sent)


def get_batch(img_objs, batch_size=64):
    data_size = len(img_objs)
    i = tf.train.range_input_producer(data_size, shuffle=False).dequeue()

    img = img_objs.img
    img_tensor = tf.convert_to_tensor(img, name='img_data', dtype=tf.float32)
    img_data = tf.strided_slice(img_tensor, [i, 0, 0, 0], [(i+1), 96, 96, 3])
    img_data = tf.reshape(img_data, [96, 96, 3])

    match_sent = img_objs.match_sent
    match_sent_tensor = tf.convert_to_tensor(
        match_sent, name='match_sent_data', dtype=tf.float32)
    match_sent_data = tf.strided_slice(match_sent_tensor, [i, 0], [(i+1), 4800])

    mismatch_sent = img_objs.mismatch_sent
    mismatch_sent_tensor = tf.convert_to_tensor(
        mismatch_sent, name='mismatch_sent_data', dtype=tf.float32)
    mismatch_sent_data = tf.strided_slice(mismatch_sent_tensor, [i, 0], [(i+1), 4800])
    img_batch, match_sent_batch, mismatch_sent_batch = tf.train.batch(
        [img_data, match_sent_data, mismatch_sent_data])

    return img_batch, match_sent_batch, mismatch_sent_batch
    
def build_imgs():
    with open('/home/newslab/MLDS2017/hw3/data/tags_clean.csv', 'r') as tag_file:
        tag_reader = csv.reader(tag_file, delimiter='\t')
        img_objs = []
        for row in tag_reader:
            img_id = row[0].split(',')[0]
            tag_row = [row[0].split(',')[1]] + row[1:]
            img = skimage.io.imread(
                '/home/newslab/MLDS2017/hw3/data/faces/{}.jpg'.format(int(img_id)))
            match_sent = []
            mismatch_sent = []
            tag_eyes = []
            tag_hair = []
            for tag in tag_row:
                tag = tag.split(':')[0]
                if 'eyes' in tag:
                    tag_eyes.append(tag)
                if 'hair' in tag:
                    tag_hair.append(tag)
            for t_h in tag_hair:
                for t_e in tag_eyes:
                    match_sent.append('{} {}'.format(t_h, t_e))
            if match_sent:
                img_objs.append(realimg(img, match_sent))
        model = skipthoughts.load_model()
        k = 0
        for img_obj1 in img_objs:
            find = 0
            for img_obj2 in img_objs[1:]:
                for sent in img_obj2.match_sent:
                    if sent not in img_obj1.match_sent:
                        img_obj1.mismatch_sent.append(sent)
                        find += 1
                    if find >= 1: break
                if find >= 1: break
            img_obj1.sent2embed(model)
            print "{}/{}".format(k, len(img_objs))
            k += 1
    with open("img_objs.pk", "w") as f:
        pk.dump(img_objs, f)
build_imgs()

