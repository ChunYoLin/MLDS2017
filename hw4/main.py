import os
import re
import sys
import logging
import time
import random
import math
import pickle as pk

import tensorflow as tf
import numpy as np
from numpy.random import multinomial

import data_readerv2
import at_seq2seq

_bucket = [(5, 10), (10, 15), (20, 25), (40, 50)]
def create_model(sess, forward_only, vocab_size):
    model = at_seq2seq.chatbot(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        buckets=_bucket,
        size=256,
        num_layers=1,
        max_gradient_norm=5.0,
        batch_size=64,
        learning_rate=0.001,
        learning_rate_decay_factor=0.99,
        use_lstm=False,
        num_samples=512,
        forward_only=forward_only,
        dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    return model

def train():
    with tf.Session() as sess:
        w_id, inv_w_id, train_set = data_readerv2.read_selected(20000)
        model = create_model(sess, False, len(w_id))
        model.load(sess, './at_s2s/')
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_bucket))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        print "-----start training-----"
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
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
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                        step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = './at_s2s/'
                model_name = './model'
                model.saver.save(
                    sess, os.path.join(checkpoint_path, model_name), 
                    model.global_step)
                step_time, loss = 0.0, 0.0
def test():
    with tf.Session() as sess:
        #  w_id, inv_w_id, train_set = data_readerv2.read_selected(0)
        print 'load word dict......'
        with open('w_id.pk', 'r') as w, open('inv_w_id.pk', 'r') as inv_w:
            w_id = pk.load(w)
            inv_w_id = pk.load(inv_w)
        model = create_model(sess, True, len(w_id))
        model.load(sess, './at_s2s/')
        model.batch_size = 1
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        def s2id(s):
            s_id = []
            for w in s.split():
                s_id.append(w_id[w])
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
            print id2s(outputs)
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
test()
