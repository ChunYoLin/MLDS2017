import os
import re
import sys
import logging
import time
import random
import math
import pickle as pk

import nltk
import tensorflow as tf
import numpy as np
from numpy.random import multinomial

import data_readerv2
#import rl_seq2seq
import at_seq2seq

tf.app.flags.DEFINE_boolean("train", True, "True for training False to testing")
tf.app.flags.DEFINE_string("input_file", None, "testing input")
tf.app.flags.DEFINE_string("output_file", None, "testing output")
tf.app.flags.DEFINE_boolean("rl", False, "True for reinforcement learning")
selected_path = '../Data/train_all.txt'
open_path = '../Data/OpenData/'
FLAGS = tf.app.flags.FLAGS

_bucket = [(5, 10), (10, 15), (20, 25), (40, 50)]
def create_model(sess, forward_only, word_dict, inv_word_dict):
    model = at_seq2seq.chatbot(
        word_dict=word_dict,
        inv_word_dict=inv_word_dict,
        source_vocab_size=len(word_dict),
        target_vocab_size=len(word_dict),
        buckets=_bucket,
        size=256,
        num_layers=1,
        max_gradient_norm=5.0,
        batch_size=64,
        learning_rate=0.001,
        learning_rate_decay_factor=0.99,
        use_lstm=True,
        num_samples=512,
        forward_only=forward_only,
        dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    return model

def train():
    with tf.Session() as sess:
        #  w_id, inv_w_id, train_set = data_readerv2.read_chatter()
        print 'load word dict......'
        with open('./HW4_Model/DICT/w_id.pk', 'r') as w, open('./HW4_Model/DICT/inv_w_id.pk', 'r') as inv_w:
            w_id = pk.load(w)
            inv_w_id = pk.load(inv_w)
        print 'finish load word dict......'
        print "Reading training Data"
        #train_set = data_readerv2.read_lines(w_id, selected_path, 20000)
        train_set = data_readerv2.read_path(w_id, open_path, 1000000, 1000)
        print "Finish reading training Data"
        model = create_model(sess, False, w_id, inv_w_id)
        model.load(sess, './HW4_Model/at_open_model/')
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_bucket))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        print "-----start training-----"
        debug = False
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            if FLAGS.rl == True:
                model.batch_size = 1
                encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
                    w_id, train_set, bucket_id, model.batch_size)

                _, step_loss, _ = model.step_rl(sess, encoder_inputs,
                                                decoder_inputs, target_weights, 
                                                bucket_id, debug=debug)
                step_time += (time.time() - start_time) / 100
                loss += step_loss / 100
                current_step += 1
                if current_step % 100 == 0:
                    debug = True
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print ("global step %d learning rate %.4f step-time %.2f loss "
                           "%f" % (model.global_step.eval(), model.learning_rate.eval(),
                            step_time, loss))
                    previous_losses.append(loss)
                    checkpoint_path = './rl_s2s_model_twitter/'
                    model_name = './model'
                    model.saver.save(
                        sess, os.path.join(checkpoint_path, model_name), 
                        model.global_step)
                    step_time, loss = 0.0, 0.0
                else:
                    debug = False
            else:
                encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
                    w_id ,train_set, bucket_id, model.batch_size)
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / 100
                loss += step_loss / 100
                current_step += 1
                if current_step % 100 == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print ("global step %d learning rate %.4f step-time %.2f loss "
                           "%f" % (model.global_step.eval(), model.learning_rate.eval(),
                            step_time, loss))
                    previous_losses.append(loss)
                    checkpoint_path = './HW4_Model/at_open_model/'
                    model_name = './HW4_Model/open_model'
                    model.saver.save(
                        sess, os.path.join(checkpoint_path, model_name), 
                        model.global_step)
                    step_time, loss = 0.0, 0.0
def test():
    with tf.Session() as sess:
        print 'load word dict......'
        with open('./HW4_Model/DICT/w_id.pk', 'r') as w, open('./HW4_Model/DICT/inv_w_id.pk', 'r') as inv_w:
            w_id = pk.load(w)
            inv_w_id = pk.load(inv_w)
        print 'finish load word dict......'
        #  train_set = data_readerv2.read_lines(w_id, './data/chat.txt', 0)
        model = create_model(sess, True, w_id, inv_w_id)
        model.load(sess, './HW4_Model/at_open_model/')
        model.batch_size = 1
        if FLAGS.output_file:
            F_out = open(FLAGS.output_file, "w")
            if not F_out :
                print "[Error] Cannot write this file ",FLAGS.output_file
                return
        if FLAGS.input_file:
            F_in = open(FLAGS.input_file, "r")
            if not F_in :
                print "[Error] Cannot read this file ",FLAGS.input_file
                return
            sentence = F_in.readline()
        else:
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
        def s2id(s):
            s_id = []
            s = nltk.word_tokenize(s.lower())
            for w in s:
                if w in w_id:
                    s_id.append(w_id[w])
                else:
                    s_id.append(0)
            return s_id
        def id2s(ids):
            s = ""
            for w in ids:
                if w != 0:
                    s += inv_w_id[w]
                    s += " "
                if w == w_id['EOS']:
                    break
            return s
        print '-----start testing-----'
        while sentence:
            token_ids = s2id(sentence)
            bucket_id = len(_bucket) - 1
            for i, bucket in enumerate(_bucket):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", sentence)
            encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
                w_id, {bucket_id: [(token_ids, [])]}, bucket_id, model.batch_size)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                              target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            #  outputs = [int(multinomial(20, logit, 1)) for logit in output_logits]
            if FLAGS.output_file:
                F_out.write(id2s(outputs)+"\n")
            else:
                print id2s(outputs)
            if FLAGS.input_file :
                sentence  = F_in.readline()
            else :
                sys.stdout.write("> ")
                sys.stdout.flush()
                sentence = sys.stdin.readline()

        if FLAGS.input_file:
            F_in.close()
        if FLAGS.output_file:
            F_out.close()
def main(_):
    if FLAGS.train:
        train()
    else:
        test()
if __name__ == "__main__":
    tf.app.run()
