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
import os
import re


image_path = '/home/chunyo/MLDS2017/hw3/data/faces/'
sent_path = '/home/chunyo/MLDS2017/hw3/data/tags_clean.csv/'


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class GAN(object):
    def __init__(self, sess, img_h, img_w, img_c):
        self.output_height, self.output_width = img_h, img_w
        self.c_dim = img_c
        self.sess = sess
        self.gf_dim = 64
        self.df_dim = 64
        self.batch_size = 64
        self.orig_sent_size = 4800
        self.sent_size = 128
        self.d_bn0 = batch_norm(name="d_bn0")
        self.d_bn1 = batch_norm(name="d_bn1")
        self.g_bn0 = batch_norm(name="g_bn0")
        self.g_bn1 = batch_norm(name="g_bn1")
        self.build_model()

    def build_model(self):
        #  Encode matching text description
        self.match_sent_highdim = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.orig_sent_size],
            name="match_sent_highdim")
        self.match_sent = self.sent_dim_reducer(self.match_sent_highdim)
        match_sent = self.match_sent
        #  Encode mis-matching text description
        self.mismatch_sent_highdim = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.orig_sent_size],
            name="mismatch_sent_highdim")
        self.mismatch_sent = self.sent_dim_reducer(
            self.mismatch_sent_highdim, reuse=True)
        mismatch_sent = self.match_sent
        #  Input real image
        image_dims = [self.output_height, self.output_width, self.c_dim]
        self.real_image = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims,
            name='real_image')
        real_image = self.real_image
        #  Draw sample of random noise
        z_dims = 100
        dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        sample_z = dist.sample([self.batch_size, z_dims])
        #  Concat the noise to match sentence
        self.G_in = tf.concat([sample_z, match_sent], 1)
        #  Forward through generator
        self.fake_image = self.generator(self.G_in)
        fake_image = self.fake_image
        #  real image, right text
        self.Sr, self.Sr_logits = self.discriminator(
            match_sent, real_image, reuse=False)
        Sr, Sr_logits = self.Sr, self.Sr_logits
        #  real image, wrong text
        self.Sw, self.Sw_logits = self.discriminator(
            mismatch_sent, real_image, reuse=True)
        Sw, Sw_logits = self.Sw, self.Sw_logits
        #  fake image, right text
        self.Sf, self.Sf_logits = self.discriminator(
            match_sent, fake_image, reuse=True)
        Sf, Sf_logits = self.Sf, self.Sf_logits
        #  loss of discriminator
        #  real image loss of discriminator
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(Sr), logits=Sr_logits))
        #  wrong sentence loss of discriminator
        self.d_loss_Sw = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(Sw), logits=Sw_logits)
        #  fake image loss of discriminator
        self.d_loss_Sf = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(Sf), logits=Sf_logits)
        #  combine discriminator loss
        self.d_loss_fake = tf.reduce_mean(
            (self.d_loss_Sw + self.d_loss_Sf) / 2.)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        #  loss of generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(Sf), logits=Sf_logits))
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
        for epoch in 1000:
            imgs_batch, match_sent_batch, mismatch_sent_batch = 
            data_reader.get_batch(img_objs, self.batch_size)

        #  input sentence
        #  input image
        #  for img in os.listdir(image_path):
            #  print img

    def sent_dim_reducer(self, sent, reuse=False):
        with tf.variable_scope("sent_dim_reducer") as scope:
            if reuse:
                scope.reuse_variables()
            w = tf.get_variable(
                "g_sent_reduce_w", [self.orig_sent_size, self.sent_size],
                tf.float32, tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable(
                "g_sent_reduce_b", [self.sent_size],
                tf.float32, initializer=tf.constant_initializer(0.0))
            embed = tf.matmul(sent, w) + b
            return tf.sigmoid(embed)

    def discriminator(self, sent, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            sent_repicate = sent
            for i in range(int(h1.shape[1])**2 - 1):
                sent_repicate = tf.concat([sent_repicate, sent], 1)
            sent_repicate = tf.reshape(
                sent_repicate,
                [self.batch_size, int(h1.shape[1]), int(h1.shape[1]), -1])

            h1 = tf.concat([h1, sent_repicate], 3)
            h2 = linear(tf.reshape(h1, [self.batch_size, -1]), 1, 'd_h1_lin')

        return tf.nn.sigmoid(h2), h2

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, (self.gf_dim*2*s_h4*s_w4), 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h4, s_w4, self.gf_dim * 2])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h2, s_w2, self.gf_dim*1],
                name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            self.h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h, s_w, self.c_dim],
                name='g_h2', with_w=True)
            return tf.nn.tanh(self.h2)

sess = tf.Session()
a = GAN(sess, 96, 96, 3)
a.train()
