import os
import re
import random
import pickle as pk

import tensorflow as tf
import numpy as np
import nltk

import data_reader

tf.app.flags.DEFINE_boolean("train", True, "True for training False to testing")
FLAGS = tf.app.flags.FLAGS


class chatbot(object):

    def __init__(self, sess, ops='train'):
        self.sess = sess
        self.data_size = 20000
        self.num_steps = 50
        self.embed_size = 512
        self.num_layers = 1
        if ops == 'train':
            self.batch_size = 64
        elif ops == 'test':
            self.batch_size = 1
        (encoder_input_ids, 
        decoder_input_ids,
        decoder_target_ids,
        word_dict, 
        inv_word_dict) = data_reader.read_selected()
        self.batchs = data_reader.build_selected_batch(
            encoder_input_ids,
            decoder_input_ids,
            decoder_target_ids,
            word_dict, 
            self.batch_size)
        self.word_dict = word_dict
        self.inv_word_dict = inv_word_dict
        self.vocab_size = len(word_dict)
        self.build_model(ops)
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_op = self.global_step.assign(self.global_step + 1)
        self.saver = tf.train.Saver(tf.global_variables())
    
    def build_model(self, ops):
        self.encoder_inputs = []
        for i in range(self.num_steps): 
            self.encoder_inputs.append(tf.placeholder(
                shape=[None], dtype=tf.int32, name='encoder_inputs{0}'.format(i)))
        #  self.encoder_inputs = tf.placeholder(
            #  shape=[None, None], dtype=tf.int32, name='encoder_inputs')
        self.encoder_lengths = tf.placeholder(
            shape=[self.batch_size], dtype=tf.int32, name='encoder_lengths')
        self.decoder_targets = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='decoder_inputs')
        with tf.device('cpu:0'):
            self.embeddings = tf.get_variable(
                'embeddings', [self.vocab_size, self.embed_size], 
                dtype=tf.float32
            )
        
        self.encoder_inputs_embed = []
        for i in range(self.num_steps):
            self.encoder_inputs_embed.append(tf.nn.embedding_lookup(
                self.embeddings, self.encoder_inputs[i]))


        def single_cell():
            return tf.contrib.rnn.LSTMCell(
                self.embed_size, forget_bias=0., state_is_tuple=True)

        #---build encoder---#
        with tf.variable_scope("model") as scope:
            cell = single_cell()
            if self.num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [single_cell() for _ in range(self.num_layers)])
            #  encoder_cell = lstm_cell()
            
            encoder_out = tf.contrib.rnn.static_rnn(
                cell=cell, 
                inputs=self.encoder_inputs_embed, 
                initial_state=cell.zero_state(self.batch_size, dtype=tf.float32),
                dtype=tf.float32, 
                sequence_length=self.encoder_lengths,
                scope=scope
            )
            encoder_final_outputs = encoder_out[0]
            self.encoder_final_state = encoder_out[1]
            del encoder_final_outputs

            #---build decoder---#
            softmax_w = tf.get_variable(
                "softmax_w", [self.embed_size, self.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable(
                "softmax_b", [self.vocab_size], dtype=tf.float32)
            self.softmax_w = softmax_w
            #  with tf.variable_scope('decoder') as scope:
            decoder_outputs = []
            decoder_output_words = []
            decoder_input_real = []
            for i in range(self.num_steps):
                tf.get_variable_scope().reuse_variables()
                if i == 0:
                    embed_idx = [
                        self.word_dict["BOS"] for _ in range(self.batch_size)
                    ]
                    state = self.encoder_final_state
                else:
                    if ops == 'train':
                        embed_idx = self.decoder_inputs[:, i-1]
                        #  embed_idx = argmax_word
                    elif ops == 'test':
                        embed_idx = sample_word

                decoder_input_real.append(embed_idx)
                decoder_inputs_embed = tf.nn.embedding_lookup(
                    self.embeddings, embed_idx)
                decoder_inputs_embed = tf.reshape(
                    decoder_inputs_embed, [self.batch_size, self.embed_size])
                output, state = cell(decoder_inputs_embed, state)
                decoder_outputs.append(output)
                if ops == 'test':
                    output_logits = tf.matmul(output, softmax_w) + softmax_b
                    output_probs = tf.log(tf.nn.softmax(output_logits))
                    argmax_word = tf.to_int32(tf.argmax(output_probs, 1))
                    sample_word = tf.multinomial(output_probs, 1)
                    sample_word = tf.to_int32(
                        tf.reshape(sample_word, [self.batch_size]))
                    decoder_output_words.append(sample_word)

        decoder_outputs = tf.concat(axis=1, values=decoder_outputs)
        decoder_outputs = tf.reshape(decoder_outputs, [-1, self.embed_size])
        decoder_logits = (
            tf.matmul(decoder_outputs, softmax_w) + softmax_b)
        
        self.decoder_logits = tf.reshape(
            decoder_logits, [self.batch_size, -1, self.vocab_size])
        self.decoder_output_words = tf.argmax(self.decoder_logits, -1)
        self.decoder_input_real = tf.transpose(decoder_input_real)
        #  self.decoder_output_words = tf.transpose(decoder_output_words)
        if ops != 'train':
            return
        loss = tf.nn.sampled_softmax_loss(
            weights=tf.transpose(softmax_w), 
            biases=softmax_b,
            labels=tf.reshape(self.decoder_targets, [-1, 1]),
            inputs=decoder_outputs,
            num_sampled=64,
            num_classes=self.vocab_size
        )
        self.loss = tf.reduce_mean(loss)
        self.optim = tf.train.AdamOptimizer(0.002).minimize(self.loss)

    def train(self):
        encoder_input_batchs = self.batchs[0]
        encoder_input_lengths = self.batchs[1]
        decoder_input_batchs = self.batchs[2]
        decoder_target_batchs = self.batchs[3]
        self.sess.run(tf.global_variables_initializer())
        self.load('./basic_model/')
        #  for epoch in range(1000):
        epoch = 0
        losses = 0.
        while True:
            j = random.randint(0, len(encoder_input_batchs)-1)
            fetchs = {
                'optim': self.optim,
                'loss': self.loss,
                'in': self.decoder_input_real,
                'pred': self.decoder_output_words,
                'step': self.global_step,
                }
            feed_dict = {
                self.encoder_lengths: encoder_input_lengths[j],
                self.decoder_targets: decoder_target_batchs[j],
                self.decoder_inputs: decoder_input_batchs[j],
            }
            for i in range(self.num_steps):
                feed_dict[self.encoder_inputs[i]] = encoder_input_batchs[j][i]

            vals = self.sess.run(
                fetchs,
                feed_dict=feed_dict
            )
            global_step = vals['step']
            loss = vals['loss']
            losses += loss
            IN = vals['in'].reshape([self.batch_size, self.num_steps])
            pred = vals['pred'].reshape([self.batch_size, self.num_steps])
            self.sess.run(self.global_step_op)
            epoch += 1
            if (epoch) % 100 == 0:
                print '----------------------'
                print 'epoch {} loss {}'.format(global_step, losses/100.)
                idx = random.randint(0, self.batch_size-1)
                encoder_input_ = []
                for i in range(self.num_steps):
                    w = encoder_input_batchs[j][i][idx]
                    encoder_input_.append(w)
                print 'Encoder Input: {}'.format(self.id2s(encoder_input_))
                print 'Decoder Input: {}'.format(self.id2s(IN[idx]))
                print 'Target: {}'.format(
                    self.id2s(decoder_target_batchs[j][idx]))
                print 'Output: {}'.format(self.id2s(pred[idx]))
                losses = 0.
                self.save('./basic_model/', global_step)
        
    def test(self, s):
        encoder_input = list(reversed(self.s2id(s)))
        encoder_len = len(encoder_input)
        encoder_pad = [self.word_dict["PAD"]]*(self.num_steps-encoder_len)
        encoder_input = encoder_pad + encoder_input
        encoder_input = [[w] for w in encoder_input]
        encoder_length = [encoder_len]

        fetchs = {
            'in': self.decoder_input_real,
            'pred': self.decoder_output_words,
        }
        feed_dict = {}
        for i in range(self.num_steps):
            feed_dict[self.encoder_inputs[i]] = encoder_input[i]
        feed_dict[self.encoder_lengths] = encoder_length
        vals = self.sess.run(
            fetchs,
            feed_dict=feed_dict
        )
        IN = vals['in'].reshape([self.batch_size, self.num_steps])
        pred = vals["pred"].reshape([self.batch_size, self.num_steps])

        #  probs = vals["probs"][0, 0]
        print "Test Input: {}".format(s)
        #  print "Test real Input: {}".format(self.id2s(IN[0]))
        print "Test Output: {}".format(self.id2s(pred[0]))

    def id2s(self, ids):
        s = ""
        for w in ids:
            if w != self.word_dict['PAD']:
                s += self.inv_word_dict[w]
                s += " "
            if w == 2:
                break
        return s

    def s2id(self, s):
        s_id = []
        s = nltk.word_tokenize(s)
        for w in s:
            if w in self.word_dict:
                s_id.append(self.word_dict[w])
            else:
                s_id.append(0)
        return s_id

    def Reward(self, h, x, t):
        pass

    def save(self, checkpoint_dir, step):
        model_name = "basic.model"
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

def main(_):
    with tf.Graph().as_default(): 
        with tf.Session() as sess:
            if FLAGS.train:
                with tf.variable_scope("Model"):
                    train_model = chatbot(sess=sess, ops='train')
                    train_model.train()
            else:
                with tf.variable_scope("Model"):
                    test_model = chatbot(sess=sess, ops='test')
                    test_model.load('./basic_model/')
                    while True:
                        s = raw_input("Input: ")
                        test_model.test(s)

if __name__ == "__main__":
    tf.app.run()
