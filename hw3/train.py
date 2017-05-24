import numpy as np
import tensorflow as tf
from skip_thoughts import skipthoughts
import scipy.misc
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from ops import *
import data_reader
from data_reader import realimg
import os
import re
import cPickle as pk


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class WGAN(object):
    def __init__(self, sess, img_h, img_w, img_c, op):
        #---input setting---#
        self.sess = sess
        self.op = op
        self.output_height, self.output_width = img_h, img_w
        self.c_dim = img_c
        self.orig_embed_size = 4800
        #---training data---#
        if op == "train":
            self.batch_size = 64
            print "loading training data......"
            with open("./train_data/img_objs_new.pk", "r") as f:
                img_objs = pk.load(f)
            batch = data_reader.get_train_batch(img_objs, self.batch_size)
            self.rimg_batch = batch[0]
            self.wimg_batch = batch[1]
            self.match_embed_batch = batch[2]
            self.mismatch_embed_batch = batch[3]
        #---testing data---#
        if op == "test":
            self.batch_size = 1
            self.test_sent = tf.placeholder(tf.float32, shape=
            [1, self.orig_embed_size])
        #---model network setting---#
        self.gf_dim = 64
        self.df_dim = 64
        self.z_dim = 100
        self.embed_size = 128
        self.keep_prob = tf.placeholder(tf.float32)
        #---batch_norm of discriminator---#
        self.d_bn0 = batch_norm(name="d_bn0")
        self.d_bn1 = batch_norm(name="d_bn1")
        self.d_bn2 = batch_norm(name="d_bn2")
        self.d_bn3 = batch_norm(name="d_bn3")
        self.d_bn4 = batch_norm(name="d_bn4")
        #---batch_norm of generator---#
        self.g_bn0 = batch_norm(name="g_bn0")
        self.g_bn1 = batch_norm(name="g_bn1")
        self.g_bn2 = batch_norm(name="g_bn2")
        self.g_bn3 = batch_norm(name="g_bn3")
        #---build model---#
        print "building model......"
        self.build_model()

    def build_model(self):
        #---Prepare data tensor---#
        #  Draw sample from random noise
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #  Encode testing captions
        if self.op == "test":
            self.test_embed = self.sent_dim_reducer(self.test_sent, name='g_sent_reduce')
            self.sample_in = tf.concat([self.z, self.test_embed], 1)
            self.sample = self.sampler(self.sample_in)
            self.saver = tf.train.Saver()
            return
        #  Encode matching captions
        self.g_h = self.sent_dim_reducer(self.match_embed_batch, name='g_sent_reduce')
        self.d_h = self.sent_dim_reducer(self.match_embed_batch, name='d_sent_reduce')
        #  Encode mis-matching captions
        self.d_h_ = self.sent_dim_reducer(self.mismatch_embed_batch, name='d_sent_reduce', reuse=True)
        #  flip image horizontally
        self.rimg_batch_flip = []
        for i in range(self.rimg_batch.shape[0]):
            self.rimg_batch_flip.append(
                tf.image.random_flip_left_right(self.rimg_batch[i]))
        self.rimg_batch_flip = tf.convert_to_tensor(self.rimg_batch_flip)

        #---Forward through generator---#
        self.G_in = tf.concat([self.z, self.g_h], 1)
        self.fimg_batch = self.generator(self.G_in)
        
        #---Forward through discriminator---#
        #  real image, right text
        self.ri, self.ri_logits = self.discriminator(
            self.d_h, self.rimg_batch_flip, reuse=False)
        ri, ri_logits = self.ri, self.ri_logits
        #  fake image, right text
        self.fi, self.fi_logits = self.discriminator(
            self.d_h, self.fimg_batch, reuse=True)
        fi, fi_logits = self.fi, self.fi_logits
        #  real image, wrong text
        self.wt, self.wt_logits = self.discriminator(
            self.d_h_, self.rimg_batch_flip, reuse=True)
        wt, wt_logits = self.wt, self.wt_logits
        #  wrong image, right text
        self.wi, self.wi_logits = self.discriminator(
            self.d_h, self.wimg_batch, reuse=True)
        wi, wi_logits = self.wi, self.wi_logits
        
        #---define loss tensor---#
        #  loss of generator
        self.g_loss = tf.reduce_mean(-fi_logits)
        #  loss of discriminator
        self.d_loss = (
            tf.reduce_mean(ri_logits) -
            tf.reduce_mean(fi_logits) -
            tf.reduce_mean(wt_logits) -
            tf.reduce_mean(wi_logits)
        )
        #---seperate the variables of discriminator and generator by name---#
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()

        #---weight clipping---#
        self.d_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]

        #---training op---#
        learning_rate_d = 0.00005
        learning_rate_g = 0.00005
        self.d_optim = tf.train.RMSPropOptimizer(learning_rate_d).minimize(
            -self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.RMSPropOptimizer(learning_rate_g).minimize(
            self.g_loss, var_list=self.g_vars)

    def train(self):
        #  session
        sess = self.sess
        #  initial all variable
        sess.run(tf.global_variables_initializer())
        self.load('./wgan_new/')
        tf.train.start_queue_runners(sess)
        for epoch in range(100000):
            #  sample noise
            batch_z = np.random.uniform(
                -1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            print "--------------------------------"
            print "epoch {}".format(epoch)
            #  update discriminator parameters
            for _ in range(5):
                fetches = {
                    "d_loss": self.d_loss, 
                    "d_optim": self.d_optim, 
                    "d_clip": self.d_clip,
                }
                feed_dict = {
                    self.z: batch_z,
                    self.keep_prob: 1.0,
                }
                vals = sess.run(fetches, feed_dict=feed_dict)
                d_loss = vals["d_loss"]
            print "d_loss {}".format(d_loss)
            #  update generator parameters
            for _ in range(1):
                fetches = {
                    "real_imgs": self.rimg_batch_flip,
                    "sample_imgs": self.fimg_batch,
                    "g_loss": self.g_loss,
                    "g_optim": self.g_optim,
                }
                feed_dict = {
                    self.z: batch_z,
                    self.keep_prob: 1.0,
                }
                vals = sess.run(fetches, feed_dict=feed_dict)
                g_loss = vals["g_loss"]
                sample_imgs = vals["sample_imgs"]
            print "g_loss {}".format(g_loss)
            #  save and test the model
            if (epoch+1) % 100 == 0:
                self.save('./wgan_new/', epoch+11229)
                for idx, img in enumerate(sample_imgs):
                    skimage.io.imsave("./sample/{}.jpg".format(idx), img)
    def test(self):
        #  load model
        model_name = 'wgan_new'
        test_dir = '../data/test/{}/'.format(model_name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        self.load("./{}/".format(model_name))
        sess = self.sess
        test_sent = data_reader.get_test_sent()
        for idx, sent in enumerate(test_sent):
            sent = np.reshape(sent, (1, -1))
            for i in range(5):
                z = np.random.uniform(-1, 1, [1, self.z_dim]).astype(np.float32)
                fetches = {
                    "sample_in": self.sample_in, 
                    "sample_img": self.sample
                }
                feed_dict = {
                    self.test_sent: sent,
                    self.z: z,
                }
                vals = sess.run(fetches, feed_dict=feed_dict)
                #  write out the generated image
                sample_img = vals["sample_img"]
                skimage.io.imsave("../data/test/{}/{}_{}.jpg".format(model_name, idx+1, i+1), sample_img)

    def sent_dim_reducer(self, sent, name, reuse=False):
        with tf.variable_scope("sent_dim_reducer") as scope:
            if reuse:
                scope.reuse_variables()
            w = tf.get_variable(
                "{}_w".format(name), [self.orig_embed_size, self.embed_size],
                tf.float32, tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable(
                "{}_b".format(name), [self.embed_size],
                tf.float32, initializer=tf.constant_initializer(0.0))
            embed = tf.matmul(sent, w) + b
            return embed

    def discriminator(self, sent, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            print "Discriminator"
            if reuse:
                scope.reuse_variables()
            print image.shape
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            print h0.shape
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            print h1.shape
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            print h2.shape
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            print h3.shape
            sent_repicate = sent
            for i in range(int(h3.shape[1])**2 - 1):
                sent_repicate = tf.concat([sent_repicate, sent], 1)
            sent_repicate = tf.reshape(
                sent_repicate,
                [self.batch_size, int(h3.shape[1]), int(h3.shape[1]), -1])
            h3 = tf.concat([h3, sent_repicate], 3)
            print h3.shape
            h4 = lrelu(self.d_bn4(conv2d(
                h3, self.df_dim*8, 1, 1, 1, 1, name = "d_h4_conv")))
            print h4.shape
            h5 = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h4_lin')
            print h5.shape

        return tf.nn.sigmoid(h5), h5

    def generator(self, z, reuse=False):
        print "Generator"
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, (s_h16*s_w16*self.gf_dim*8), 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            print h0.shape
            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4],
                name='g_h1', with_w=True)
            h1 = tf.nn.dropout(tf.nn.relu(self.g_bn1(self.h1)), self.keep_prob)
            print h1.shape
            self.h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2],
                name='g_h2', with_w=True)
            h2 = tf.nn.dropout(tf.nn.relu(self.g_bn2(self.h2)), self.keep_prob)
            print h2.shape
            self.h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1],
                name='g_h3', with_w=True)
            h3 = tf.nn.dropout(tf.nn.relu(self.g_bn3(self.h3)), self.keep_prob)
            print h3.shape
            self.h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim],
                name='g_h4', with_w=True)
            print self.h4.shape
            return tf.nn.tanh(self.h4)

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            #  scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, (s_h16*s_w16*self.gf_dim*8), 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=False))

            self.h1 = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=False))

            self.h2 = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(self.h2, train=False))

            self.h3 = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(self.h3, train=False))

            self.h4 = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.tanh(self.h4)

    def save(self, checkpoint_dir, step):
        model_name = "basic_all.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            self.sess, os.path.join(checkpoint_dir, model_name),
            global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

sess = tf.Session()
#  train_model = GAN(sess, 64, 64, 3, "train")
#  train_model.train()
test_model = GAN(sess, 64, 64, 3, "test")
test_model.test()

