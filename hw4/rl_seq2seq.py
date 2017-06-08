import os
import re
import time
import random
import math
import pickle as pk

import nltk
import tensorflow as tf
import numpy as np
import seq2seq as tf_seq2seq

import data_readerv2


class chatbot(object):

    def __init__(self,
            word_dict,
            inv_word_dict,
            source_vocab_size,
            target_vocab_size,
            buckets,
            size,
            num_layers,
            max_gradient_norm,
            batch_size,
            learning_rate,
            learning_rate_decay_factor,
            use_lstm=False,
            num_samples=512,
            forward_only=False,
            dtype=tf.float32):

        self.word_dict = word_dict
        self.inv_word_dict = inv_word_dict
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.dummy_dialogs = ["I don't know.", "Okay.", "What?"]

        output_projection = None
        softmax_loss_function = None
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size),
                    dtype)
            softmax_loss_function = sampled_loss

        def single_cell():
            return tf.contrib.rnn.GRUCell(size)
        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], 
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                      name="weight{0}".format(i)))
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]
        if forward_only:
            self.outputs, self.losses, self.encoder_state = tf_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + 
                        output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses, self.encoder_state = tf_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + 
                        output_projection[1]
                        for output in self.outputs[b]
                    ]
        params = tf.trainable_variables()

        self.gradient_norms = []
        self.updates = []
        self.advantage = [tf.placeholder(tf.float32, name="advantage_%i" % i) 
                          for i in xrange(len(buckets))]
        opt = tf.train.AdamOptimizer(self.learning_rate)
        for b in xrange(len(buckets)):
            self.losses[b] = tf.subtract(self.losses[b], self.advantage[b])
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())
        
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, 
             bucket_id, forward_only, advantage=None):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        input_feed = {}
        for l in xrange(len(self.buckets)):
            input_feed[self.advantage[l].name] = advantage[l] if advantage else 0
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
            self.gradient_norms[bucket_id],  # Gradient norm.
            self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.encoder_state[bucket_id], 
                           self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2:]  # No gradient norm, loss, outputs.

    def step_rl(self, session, encoder_inputs, decoder_inputs, target_weights,
               bucket_id, debug=True):
        # initialize 
        init_inputs = [encoder_inputs, decoder_inputs, target_weights, bucket_id]
        sent_max_length = self.buckets[-1][0] - 1
        input_tokens, input_txt = self.logits2tokens(encoder_inputs, sent_max_length, reverse=True)
        out_txt = " ".join(input_txt)
        if debug: 
            print "-----------dialogs begin------------"
            print("[INPUT]: %s" %out_txt)
        encoder_state, step_loss, output_logits = self.step(
            session, encoder_inputs, decoder_inputs, target_weights,
            bucket_id, forward_only=True)
        resp_tokens, resp_txt = self.logits2tokens(output_logits, sent_max_length)
        out_txt = " ".join(resp_txt)
        if debug: 
            print("[RESP]: (%.4f) %s"%(step_loss, out_txt))
        for i, bucket in enumerate(self.buckets):
            if bucket[0] >= len(resp_tokens):
                bucket_id = i
                break
        feed_data = {bucket_id: [(resp_tokens, [])]}
        encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
            self.word_dict, feed_data, bucket_id, 1)
        r1 = []
        for d in self.dummy_dialogs:
            d_tokens = self.string2tokens(d, sent_max_length)
            r1.append(self.logProb(
                session, self.buckets, resp_tokens, d_tokens))
        r1 = -np.mean(r1) if r1 else 0

        # r3: Semantic Coherence
        r3 = -self.logProb(
                session, self.buckets, resp_tokens, input_tokens)
        rewards = 0.5*r1 + 0.5*r3
        advantage = [rewards] * len(self.buckets)
        _, step_loss, _ = self.step(session, init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3],
                  forward_only=False, advantage=advantage)
        if debug: 
            print("[Rewards]: %f"%(rewards))
        # Initialize
        #  ep_rewards, ep_step_loss, enc_states = [], [], []
        #  ep_encoder_inputs, ep_target_weights, ep_bucket_id = [], [], []

        # [Episode] per episode = n steps, until break
        #  while False:
            #  #----[Step]----------------------------------------
            #  resp_tokens, resp_txt = self.logits2tokens(encoder_inputs, sent_max_length, reverse=True)
            #  out_txt = " ".join(resp_txt)
            #  if debug: print("[INPUT]: %s" %out_txt)
            #  encoder_state, step_loss, output_logits = self.step(
                #  session, encoder_inputs, decoder_inputs, target_weights,
                #  bucket_id, forward_only=True)

            #  # memorize inputs for reproducing curriculum with adjusted losses
            #  ep_encoder_inputs.append(encoder_inputs)
            #  ep_target_weights.append(target_weights)
            #  ep_bucket_id.append(bucket_id)
            #  ep_step_loss.append(step_loss)
            #  enc_states_vec = np.reshape(np.squeeze(encoder_state, axis=0), (-1))
            #  enc_states.append(enc_states_vec)

            #  # process response
        
            #  resp_tokens, resp_txt = self.logits2tokens(output_logits, sent_max_length)
            #  out_txt = " ".join(resp_txt)

            #  # prepare for next dialogue
            #  for i, bucket in enumerate(self.buckets):
                #  if bucket[0] >= len(resp_tokens):
                    #  bucket_id = i
                    #  break
            #  feed_data = {bucket_id: [(resp_tokens, [])]}
            #  encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
                #  self.word_dict, feed_data, bucket_id, 1)

            #  #----[Reward]----------------------------------------
            #  # r1: Ease of answering
            #  r1 = []
            #  for d in self.dummy_dialogs:
                #  d_tokens = self.string2tokens(d, sent_max_length)
                #  r1.append(self.logProb(
                    #  session, self.buckets, resp_tokens, d_tokens))
            #  r1 = -np.mean(r1) if r1 else 0

            #  # r2: Information Flow
            #  if len(enc_states) < 2:
                #  r2 = 0
            #  else:
                #  vec_a, vec_b = enc_states[-2], enc_states[-1]
                #  r2 = sum(vec_a*vec_b) / sum(abs(vec_a)*abs(vec_b))
                #  if r2 > 0:
                    #  r2 = -math.log(r2)
                #  else:
                    #  r2 = 0
            #  r2 = 0

            #  # r3: Semantic Coherence
            #  r3 = -self.logProb(
                    #  session, self.buckets, resp_tokens, ep_encoder_inputs[-1])

            #  # Episode total reward
            #  R = 0.5*r1 + 0.25*r2 + 0.5*r3
            #  if debug: 
                #  print("[RESP]: (%.4f) %s [Reward]%f"%(step_loss, out_txt, R))

            #  ep_rewards.append(R)
            #  #----------------------------------------------------
            #  if (resp_txt in self.dummy_dialogs) or (len(resp_tokens) <= 3) or (
                    #  encoder_inputs in ep_encoder_inputs): 
                #  break # check if dialog ended
          
        # gradient decent according to batch rewards
        #  rto = (max(ep_step_loss) - min(ep_step_loss)) / (max(ep_rewards) - min(ep_rewards))
        #  #  advantage = [np.mean(ep_rewards)*rto] * len(self.buckets)
        #  advantage = [np.mean(ep_rewards)] * len(self.buckets)
        #  _, step_loss, _ = self.step(session, init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3],
                  #  forward_only=False, advantage=advantage)
        
        return None, step_loss, None

    # log(P(b|a)), the conditional likelyhood
    def logProb(self, session, buckets, tokens_a, tokens_b):

        def softmax(x):
            x += 0.0000001
            total = np.sum(np.exp(x), axis=0)
            if total == 0:
                return 0
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        # prepare for next dialogue
        for i, bucket in enumerate(self.buckets):
            if bucket[0] >= len(tokens_a):
                bucket_id = i
                break
        feed_data = {bucket_id: [(tokens_a, tokens_b)]}
        encoder_inputs, decoder_inputs, target_weights = data_readerv2.get_batch(
            self.word_dict, feed_data, bucket_id, 1)
        # step
        _, _, output_logits = self.step(session, encoder_inputs, 
                                        decoder_inputs, target_weights,
                                        bucket_id, forward_only=True)
        # p = log(P(b|a)) / N
        p = 1
        for t, logit in zip(tokens_b, output_logits):
            prob = softmax(logit[0])[t]
            p *= prob
        if p != 0:
            p = math.log(p) / len(tokens_b)
        return p

    def logits2tokens(
            self, logits, sent_max_length=None, reverse=False):

        if reverse:
            tokens = [t[0] for t in reversed(logits)]
        else:
            tokens = [int(np.argmax(t, axis=1)) for t in logits]

        if self.word_dict['EOS'] in tokens:
            eos = tokens.index(self.word_dict['EOS'])
            tokens = tokens[:eos]
        txt = [self.inv_word_dict[t] for t in tokens if t is not self.word_dict['PAD']]
        if sent_max_length:
            tokens, txt = tokens[:sent_max_length], txt[:sent_max_length]
        return tokens, txt
    
    def string2tokens(self, s, sent_max_length=None):
        s = nltk.word_tokenize(s.lower())
        tokens = []
        for w in s:
            if w in self.word_dict:
                tokens.append(self.word_dict[w])
            else:
                tokens.append(0)
        return tokens

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



