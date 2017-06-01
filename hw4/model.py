import tensorflow as tf
import data_reader


class chatbot(object):

    def __init__(self):
        self.data_size = 10000
        self.batch_size = 32
        self.num_steps = 5
        self.vocab_size = 10003
        self.embed_size = 256
        self.build_model()
        self.train()
    
    def build_model(self):
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
        self.decoder_inputs_embed = tf.nn.embedding_lookup(
            self.embeddings, self.decoder_inputs)
        
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(
                self.embed_size, forget_bias=0., state_is_tuple=True)

        #---build encoder---#
        encoder_cell = lstm_cell()
        (encoder_final_outputs, self.encoder_final_state) = tf.nn.dynamic_rnn(
            encoder_cell, self.encoder_inputs_embed, 
            dtype=tf.float32, time_major=False, scope="encoder"
        )

        del encoder_final_outputs

        #---build decoder---#
        decoder_cell = lstm_cell()
        decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, self.decoder_inputs_embed,
            initial_state=self.encoder_final_state,
            dtype=tf.float32, time_major=False, scope="decoder"
        )
        self.decoder_outputs = tf.reshape(decoder_outputs, [-1, self.embed_size])
        self.softmax_w = tf.get_variable(
            "softmax_w", [self.embed_size, self.vocab_size], dtype=tf.float32)
        self.softmax_b = tf.get_variable(
            "softmax_b", [self.vocab_size], dtype=tf.float32)
        decoder_logits = (
            tf.matmul(self.decoder_outputs, self.softmax_w) + self.softmax_b)
        self.decoder_logits = tf.reshape(
            decoder_logits, [self.batch_size, -1, self.vocab_size])
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
        #  self.decoder_logits = tf.contrib.layers.linear(
            #  decoder_outputs, self.vocab_size)
        #  self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
        loss = tf.nn.sampled_softmax_loss(
            weights=tf.transpose(self.softmax_w), 
            biases=self.softmax_b,
            labels=tf.reshape(self.decoder_targets, [-1, 1]),
            inputs=self.decoder_outputs,
            num_sampled=64,
            num_classes=self.vocab_size
        )
        self.loss = tf.reduce_mean(loss)
        self.optim = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def train(self):
        convs2id, word_dict, inv_word_dict = data_reader.read_raw()
        self.word_dict = word_dict
        self.inv_word_dict = inv_word_dict
        self.batchs = data_reader.build_batch(convs2id, word_dict, 
            batch_size=self.batch_size, data_size=self.data_size)
        encoder_input_batchs = self.batchs[0]
        decoder_input_batchs = self.batchs[1]
        decoder_target_batchs = self.batchs[2]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            losses = 0.
            for j in range(len(encoder_input_batchs)):
                feed_dict = {
                    self.encoder_inputs: encoder_input_batchs[j],
                    self.decoder_targets: decoder_target_batchs[j],
                    self.decoder_inputs: decoder_input_batchs[j]
                }
                _, loss, pred = self.sess.run(
                    [self.optim, self.loss, self.decoder_prediction], 
                    feed_dict=feed_dict
                )
                losses += loss / len(encoder_input_batchs)
            print "epoch {} {}".format(epoch, losses)
            print self.id2s(pred[0])
            print self.id2s(decoder_target_batchs[j][0])
        
    #  def test(self, s):
        #  encoder_input = tf.convert_to_tensor(self.s2id(s))
        #  encoder_input = tf.reshape(encoder_input, [1, -1])
        #  self.sess.run(self.decoder_prediction)

    def id2s(self, ids):
        s = []
        for w in ids:
            if w != 0:
                s.append(self.inv_word_dict[w])
        return s

    def s2id(self, s):
        s_id = []
        for w in s:
            s_id.append(self.word_dict[w])
        return s_id

    def Reward(self):
        pass
chatbot()
