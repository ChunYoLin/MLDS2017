import tensorflow as tf


class chatbot(object):

    def __init__(self):
        self.batch_size = 1
        self.num_steps = 5
        self.vocab_size = 100
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
        batch = [[6, 0, 0], [3, 4, 0], [9, 8, 7]]
        target_batch = [[1, 2, 0], [5, 6, 0], [9, 8, 1]]
        decoder_inputs_batch = [[0, 1, 0], [0, 5, 0], [0, 9, 8]]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for i in range(100):
            for j in range(len(batch)):
                feed_dict = {
                    self.encoder_inputs: batch,
                    self.decoder_targets: target_batch,
                    self.decoder_inputs: decoder_inputs_batch
                    }
                _, loss = self.sess.run([self.optim, self.loss], feed_dict=feed_dict)
                print loss
        

    def Reward(self):
        pass
chatbot()
