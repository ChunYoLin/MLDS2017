import os
import re
import random
import tensorflow as tf
import numpy as np
import data_reader


class chatbot(object):

    def __init__(self, sess, ops='train'):
        self.sess = sess
        self.data_size = 10000
        self.num_steps = 5
        self.vocab_size = 50003
        self.embed_size = 256
        convs2id, word_dict, inv_word_dict = data_reader.read_raw()
        if ops == 'train':
            self.batch_size = 32
            self.batchs = data_reader.build_batch(convs2id, word_dict, 
                batch_size=self.batch_size, data_size=self.data_size)
        elif ops == 'test':
            self.batch_size = 1
        self.convs2id = convs2id
        self.word_dict = word_dict
        self.inv_word_dict = inv_word_dict
        self.build_model(ops)
    
    def build_model(self, ops):
        self.encoder_inputs = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='decoder_inputs')
        with tf.device('cpu:0'):
            self.embeddings = tf.get_variable(
                'embeddings', [self.vocab_size, self.embed_size], 
                dtype=tf.float32
            )
        
        self.encoder_inputs_embed = tf.nn.embedding_lookup(
            self.embeddings, self.encoder_inputs)

        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(
                self.embed_size, forget_bias=0., state_is_tuple=True)

        #---build encoder---#
        with tf.variable_scope("encoder") as scope:
            encoder_cell = lstm_cell()
            (encoder_final_outputs, self.encoder_final_state) = tf.nn.dynamic_rnn(
                encoder_cell, self.encoder_inputs_embed, 
                dtype=tf.float32, time_major=False, scope=scope
            )
        del encoder_final_outputs

        #---build decoder---#
        softmax_w = tf.get_variable(
            "softmax_w", [self.embed_size, self.vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [self.vocab_size], dtype=tf.float32)
        with tf.variable_scope('decoder') as scope:
            decoder_cell = lstm_cell()
            decoder_outputs = []
            decoder_output_words = []
            for i in range(250):
                if i == 0:
                    embed_idx = [
                        self.word_dict["BOS"] for _ in range(self.batch_size)
                    ]
                    state = self.encoder_final_state
                else:
                    tf.get_variable_scope().reuse_variables()
                    embed_idx = sample_word

                decoder_inputs_embed = tf.nn.embedding_lookup(
                    self.embeddings, embed_idx)
                decoder_inputs_embed = tf.reshape(
                    decoder_inputs_embed, [self.batch_size, self.embed_size])

                output, state = decoder_cell(decoder_inputs_embed, state)
                decoder_outputs.append(output)
                output_logits = tf.matmul(output, softmax_w) + softmax_b
                output_probs = tf.nn.softmax(output_logits) 
                sample_word = tf.multinomial(output_probs, 1)   
                decoder_output_words.append(sample_word)

        decoder_outputs = tf.reshape(
            tf.concat(axis=1, values=decoder_outputs), [-1, self.embed_size])
        decoder_logits = (
            tf.matmul(decoder_outputs, softmax_w) + softmax_b)
        self.decoder_logits = tf.reshape(
            decoder_logits, [self.batch_size, -1, self.vocab_size])
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
        
        loss = tf.nn.sampled_softmax_loss(
            weights=tf.transpose(softmax_w), 
            biases=softmax_b,
            labels=tf.reshape(self.decoder_targets, [-1, 1]),
            inputs=decoder_outputs,
            num_sampled=64,
            num_classes=self.vocab_size
        )
        self.loss = tf.reduce_mean(loss)
        self.optim = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.saver = tf.train.Saver()

    def train(self):
        encoder_input_batchs = self.batchs[0]
        decoder_input_batchs = self.batchs[1]
        decoder_target_batchs = self.batchs[2]
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1000):
            losses = 0.
            for j in range(len(encoder_input_batchs)):
                fetchs = {
                    'optim': self.optim,
                    'loss': self.loss,
                    'pred': self.decoder_prediction,
                }
                feed_dict = {
                    self.encoder_inputs: encoder_input_batchs[j],
                    self.decoder_targets: decoder_target_batchs[j],
                    self.decoder_inputs: decoder_input_batchs[j]
                }
                vals = self.sess.run(
                    fetchs,
                    feed_dict=feed_dict
                )
                loss = vals['loss']
                pred = vals['pred']
                losses += loss / len(encoder_input_batchs)
            print '----------------------'
            print 'epoch {} {}'.format(epoch, losses)
            idx = random.randint(0, self.batch_size-1)
            print 'Input: {}'.format(self.id2s(encoder_input_batchs[j][idx]))
            print 'Target: {}'.format(self.id2s(decoder_target_batchs[j][idx]))
            print 'Output: {}'.format(self.id2s(pred[idx]))
        self.save('./basic_model/', epoch)
        
    def test(self, s):
        self.load('./basic_model/')
        encoder_input = np.asarray(self.s2id(s)).reshape(1, -1)
        fetchs = {
            'pred': self.decoder_prediction,
        }
        feed_dict = {
            self.encoder_inputs: encoder_input,
        }
        vals = self.sess.run(
            fetchs,
            feed_dict=feed_dict
        )
        pred = vals["pred"]
        print "Test Input: {}".format(s)
        print "Test Output: {}".format(self.id2s(pred[0]))

    def id2s(self, ids):
        s = ""
        for w in ids:
            if w != 0:
                s += self.inv_word_dict[w]
                s += " "
        return s

    def s2id(self, s):
        s_id = []
        for w in s.split():
            s_id.append(self.word_dict[w])
        return s_id

    def Reward(self):
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

with tf.Graph().as_default(): 
    sess = tf.Session()
    with tf.variable_scope("Model"):
        train_model = chatbot(sess=sess, ops='train')
        train_model.train()
    #  with tf.variable_scope("Model"):
        #  test_model = chatbot(sess=sess, ops='test')
        #  test_model.test("fuck you bitch !")
