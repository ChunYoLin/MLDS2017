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


image_path = '/home/chunyo/MLDS2017/hw3/data/faces/'
sent_path = '/home/chunyo/MLDS2017/hw3/data/tags_clean.csv/'


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class GAN(object):
    def __init__(self, sess, img_h, img_w, img_c):
        #  input
        self.sess = sess
        self.output_height, self.output_width = img_h, img_w
        self.c_dim = img_c
        #  network setting
        self.gf_dim = 64
        self.df_dim = 64
        self.batch_size = 8
        self.orig_embed_size = 4800
        self.embed_size = 128
        #  batch_norm of discriminator
        self.d_bn0 = batch_norm(name="d_bn0")
        self.d_bn1 = batch_norm(name="d_bn1")
        self.d_bn2 = batch_norm(name="d_bn2")
        self.d_bn3 = batch_norm(name="d_bn3")
        self.d_bn4 = batch_norm(name="d_bn4")
        self.d_bn5 = batch_norm(name="d_bn5")
        self.d_bn6 = batch_norm(name="d_bn6")
        #  batch_norm of generator 
        self.g_bn0 = batch_norm(name="g_bn0")
        self.g_bn1 = batch_norm(name="g_bn1")
        self.g_bn2 = batch_norm(name="g_bn2")
        self.g_bn3 = batch_norm(name="g_bn3")
        self.g_bn4 = batch_norm(name="g_bn4")
        self.g_bn5 = batch_norm(name="g_bn5")
        #  input batch
        self.match_sent = []
        print "loading training data......"
        with open("img_objs_64.pk", "r") as f:
            img_objs = pk.load(f)
        for img in img_objs:
            for sent in img.match_sent:
                self.match_sent.append(sent)
        #  img_objs = img_objs[:12800]
        self.data_size = len(img_objs)
        print "number of image {}".format(self.data_size)
        self.batch_num = self.data_size / self.batch_size
        print "number of batch {}".format(self.batch_num)
        batch = data_reader.get_batch(img_objs, self.batch_size)
        self.img_batch = batch[0]
        self.match_embed_batch = batch[1]
        self.mismatch_embed_batch = batch[2]
        #  build model
        print "building model......"
        self.build_model()

    def build_model(self):
        #  Encode matching text description
        self.h = self.sent_dim_reducer(self.match_embed_batch)
        #  Encode mis-matching text description
        self.h_ = self.sent_dim_reducer(self.mismatch_embed_batch, reuse=True)
        #  Draw sample of random noise
        z_dims = 100
        dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        self.z = dist.sample([self.batch_size, z_dims])
        self.G_in = tf.concat([self.z, self.h], 1)
        #  Forward through generator
        self.fake_image = self.generator(self.G_in)
        self.sample = self.sampler(self.G_in)
        #  real image, right text
        self.Sr, self.Sr_logits = self.discriminator(
            self.h, self.img_batch, reuse=False)
        Sr, Sr_logits = self.Sr, self.Sr_logits
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(Sr), logits=Sr_logits))
        #  real image, wrong text
        self.Sw, self.Sw_logits = self.discriminator(
            self.h_, self.img_batch, reuse=True)
        Sw, Sw_logits = self.Sw, self.Sw_logits
        self.d_loss_Sw = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(Sw), logits=Sw_logits))
        #  fake image, right text
        self.Sf, self.Sf_logits = self.discriminator(
            self.h, self.fake_image, reuse=True)
        Sf, Sf_logits = self.Sf, self.Sf_logits
        self.d_loss_Sf = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(Sf), logits=Sf_logits))
        #  loss of discriminator
        self.d_loss_fake = (self.d_loss_Sw + self.d_loss_Sf) / 2.
        self.d_loss = self.d_loss_real + self.d_loss_fake
        #  loss of generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(Sf), logits=Sf_logits))

        #  self.d_loss = -1 * tf.reduce_mean(tf.log(Sr) + (tf.log(1-Sw) + tf.log(1-Sf) / 2.))
        #  self.g_loss = -1 * tf.reduce_mean(tf.log(Sf))
        #  seperate the variables of discriminator and generator by name
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def train(self):
        #  training op
        learning_rate = 0.0002
        d_optim = tf.train.AdamOptimizer(learning_rate).minimize(
            self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate).minimize(
            self.g_loss, var_list=self.g_vars)
        #  session
        sess = self.sess
        #  initial all variable
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess)
        for epoch in range(1000):
            for batch in range(self.batch_num):
                print "epoch {} batch {}/{}".format(epoch, batch + 1, self.batch_num)
                d_loss, _, Sr, Sw, Sf = sess.run([self.d_loss, d_optim, self.Sr, self.Sw, self.Sf])
                g_loss, _ = sess.run([self.g_loss, g_optim])
                g_loss, _ = sess.run([self.g_loss, g_optim])
                print "d_loss {}".format(d_loss)
                print "g_loss {}".format(g_loss)
                print "Sr: {}, Sw: {}, Sf: {}".format(np.mean(Sr), np.mean(Sw), np.mean(Sf))
            if (epoch+1) % 10 == 0:
                with open("./sample/match_sent/sample_sent.txt", "w") as f:
                    for batch in range(self.batch_num):
                        sample_imgs = sess.run(self.sample)
                        for img_idx, img in enumerate(sample_imgs):
                            idx = batch * self.batch_size + img_idx
                            skimage.io.imsave("./sample/{}.jpg".format(idx), img)
                            f.write("{}: {}\n".format(idx, self.match_sent[idx]))

    def sent_dim_reducer(self, sent, reuse=False):
        with tf.variable_scope("sent_dim_reducer") as scope:
            if reuse:
                scope.reuse_variables()
            w = tf.get_variable(
                "g_sent_reduce_w", [self.orig_embed_size, self.embed_size],
                tf.float32, tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable(
                "g_sent_reduce_b", [self.embed_size],
                tf.float32, initializer=tf.constant_initializer(0.0))
            embed = tf.matmul(sent, w) + b
            return tf.nn.relu(embed)

    def discriminator(self, sent, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, name='d_h4_conv')))
            h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim*32, name='d_h5_conv')))
            sent_repicate = sent
            for i in range(int(h5.shape[1])**2 - 1):
                sent_repicate = tf.concat([sent_repicate, sent], 1)
            sent_repicate = tf.reshape(
                sent_repicate,
                [self.batch_size, int(h5.shape[1]), int(h5.shape[1]), -1])
            h5 = tf.concat([h5, sent_repicate], 3)
            h6 = lrelu(self.d_bn6(conv2d(
                h5, self.df_dim*32, 1, 1, 1, 1, name = "d_h6_conv")))
            h7 = linear(tf.reshape(h5, [self.batch_size, -1]), 1, 'd_h5_lin')

        return tf.nn.sigmoid(h7), h7

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, (s_h64*s_w64*self.gf_dim*32), 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h64, s_w64, self.gf_dim*32])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h32, s_w32, self.gf_dim*16],
                name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            self.h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h16, s_w16, self.gf_dim*8],
                name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(self.h2))

            self.h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h8, s_w8, self.gf_dim*4],
                name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(self.h3))

            self.h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h4, s_w4, self.gf_dim*2],
                name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(self.h4))

            self.h5, self.h5_w, self.h5_b = deconv2d(
                h4, [self.batch_size, s_h2, s_w2, self.gf_dim*1],
                name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(self.h5))

            self.h6, self.h6_w, self.h6_b = deconv2d(
                h5, [self.batch_size, s_h, s_w, self.c_dim],
                name='g_h6', with_w=True)

            return tf.nn.tanh(self.h6)

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, (s_h64*s_w64*self.gf_dim*32), 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h64, s_w64, self.gf_dim*32])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=False))

            self.h1 = deconv2d(
                h0, [self.batch_size, s_h32, s_w32, self.gf_dim*16], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=False))

            self.h2 = deconv2d(
                h1, [self.batch_size, s_h16, s_w16, self.gf_dim*8], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(self.h2, train=False))

            self.h3 = deconv2d(
                h2, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(self.h3, train=False))

            self.h4 = deconv2d(
                h3, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(self.h4, train=False))

            self.h5 = deconv2d(
                h4, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h5')
            h5 = tf.nn.relu(self.g_bn5(self.h5, train=False))

            self.h6 = deconv2d(
                h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6')

            return tf.nn.tanh(self.h6)

sess = tf.Session()
a = GAN(sess, 96, 96, 3)
a.train()
