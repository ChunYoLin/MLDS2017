import numpy as np
import tensorflow as tf
from skip_thoughts import skipthoughts
import scipy.misc
from ops import * 
import os
import re

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class GAN(object):
    def __init__(self, sess, img_h, img_w, img_c):
        self.output_height, self.output_width = img_h, img_w
        self.c_dim = img_c
        self.sess = sess
        self.gf_dim = 64
        self.df_dim = 64
        self.batch_size = 1
        self.input_sent_size = 4800
        self.sent_size = 128
        self.d_bn0 = batch_norm(name = "d_bn0")
        self.d_bn1 = batch_norm(name = "d_bn1")
        self.g_bn0 = batch_norm(name = "g_bn0")
        self.g_bn1 = batch_norm(name = "g_bn1")
    
    def build_model(self):
        #  sample noise
        noise_dims = 100
        image_dims = [self.output_height, self.output_width]
        dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        noise = dist.sample([self.batch_size, noise_dims])
        #  reduce input sent to dim 128
        self.sent_highdim = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.input_sent_size], name="input_sent_high_dim")
        self.input_sent = self.sent_dim_reducer(self.sent_highdim)
        self.input_image = tf.placeholder(
                tf.float32, [self.batch_size] + image_dims, name='real_image')
        input_sent = self.input_sent
        input_image = self.input_image
        self.g_in = tf.concat([noise, input_sent], 1)
        self.g_out = self.generator(self.g_in)
        self.d_out_sigmoid, self.d_out = self.discriminator(self.g_out)
    def train(self):
        model = skipthoughts.load_model()
        vecs = skipthoughts.encode(model, ['blue eyes'])
        self.sess.run(tf.initialize_all_variables())
        g_img = sess.run(self.g_out, feed_dict={self.sent_highdim: vecs})

    def test(self):
        model = skipthoughts.load_model()
        vecs = skipthoughts.encode(model, ['blue eyes'])
        self.sess.run(tf.initialize_all_variables())
        g_img = sess.run(self.g_out, feed_dict={self.sent_highdim: vecs})
        for img in g_img:
            print img.shape
            scipy.misc.imsave('sample_img.jpg', img)
    
    def sent_dim_reducer(self, sent):
        with tf.variable_scope("sent_dim_reducer") as scope:
            w = tf.get_variable("w", [self.input_sent_size, self.sent_size], tf.float32,
                    tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable("b", [self.sent_size], tf.float32,
                    initializer=tf.constant_initializer(0.0))
            embed = tf.matmul(sent, w) + b
            return tf.sigmoid(embed)
    
    def discriminator(self, image):
        with tf.variable_scope("discriminator") as scope:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = linear(tf.reshape(h1, [self.batch_size, -1]), 1, 'd_h1_lin')

        return tf.nn.sigmoid(h2), h2
            
    
    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim*2*s_h4*s_w4, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h4, s_w4, self.gf_dim * 2])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            
            self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            self.h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h, s_w, self.c_dim], name='g_h2', with_w=True)
            return tf.nn.tanh(self.h2)

    def sampler():
        pass

    def text_encoder():
        pass
sess = tf.Session()
a = GAN(sess, 96, 96, 3)
a.build_model()
a.test()


