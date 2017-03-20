import tensorflow as tf
import numpy as np
import gensim
import os
import word2vec
import csv

batch_size = 1
n_inputs = 100 
n_steps = 30
n_hidden_units = 256 

model = gensim.models.Word2Vec.load('./word2vec_model')

words = word2vec.build_wordset()
new_sentences, words = word2vec.sentence_formulation(words, n_steps)
new_sentences, dictionary, inv_dictionary = word2vec.build_dataset(words, new_sentences)

vocabulary_size = len(dictionary)


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None, n_steps])
keep_prob = tf.placeholder(tf.float32)

#define weights
weights = {
    # (100, 256)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (256, vocab_size)
    'out': tf.Variable(tf.random_normal([n_hidden_units, vocabulary_size]))
}
biases = {
    # (256, )
    'in': tf.Variable(tf.constant(0.1, shape = [n_hidden_units, ])),
    # (vocab_size, )
    'out': tf.Variable(tf.constant(0.1, shape = [vocabulary_size, ]))
}


def RNN(X, weights, biases, batch_size):
    global n_hidden_units
    # X ==> (20 batch * max_len steps,  100 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X = tf.nn.dropout(X, keep_prob)
    # into hidden
    # X_in = (20 batch *  n_steps, 256 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (20 batch, n_steps, 256 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,
                                             forget_bias = 0.0,
                                             state_is_tuple = True)

    # lstm cell is divided into two parts (c_state, h_state)

    init_state = lstm_cell.zero_state(batch_size, dtype = tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,
                                             X_in,
                                             initial_state = init_state)
                                             #time_major = False)
    outputs = tf.reshape(outputs,[-1, n_hidden_units])
    logits = tf.matmul(outputs, weights['out']) + biases['out']
    probs = tf.nn.softmax(logits)
    #prob to predict next vocab  
    return logits, probs, lstm_cell, init_state, final_state

answer = []

def predict():
    global answer
    global batch_size
    global n_steps
    global n_inputs
    max_len = 0
    logits, probs, lstm_cell, init_state, final_state = RNN(x, weights, biases, batch_size) 
    
    file = open('./testing_data.csv', 'r')
    reader = csv.reader(file)
    next(reader, None)
    
    with tf.Session() as sess:    
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('training_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        sess.run(tf.global_variables_initializer())
        
        state_ = sess.run(lstm_cell.zero_state(1, tf.float32))
        
        max_line = 0
        for row in reader:
            option = row[2:7]
            question = row[1].replace('.', '')             \
                             .replace('?', ' ')            \
                             .replace(',', ' ')            \
                             .replace('"', ' ')            \
                             .replace('!', ' ')            \
                             .replace(':', ' ')            \
                             .replace(';', ' ')            \
                             .replace('(', ' ')            \
                             .replace(')', ' ')            \
                             .replace('*', ' ')            \
                             .split()
            
            for i in range(len(question)):
                if question[i] == '_____':
                    question_word_idx = i
                    question = question[0: i]
                    break

            if max_len < len(question):
                max_len = len(question)    
            
            now_len = n_steps - len(question) 
            while now_len > 0:
                question.append('UNK')
                now_len -= 1 

            embd_question = np.ndarray([1, n_steps, n_inputs], dtype = np.float32)
            for i in range(n_steps):
                try:
                    embd_question[0, i] = model[question[i]]
                except KeyError:
                    #print "not found! ",  question[i]
                    embd_question[0, i] = model['UNK']
            test_prob, state_ = sess.run([probs, init_state], 
                                         feed_dict = {x: embd_question, 
                                                      keep_prob: 1, 
                                                      init_state: state_})
            print(row[0])
            prob_arr = []
            for i in range(5):
                #if option[i] not in dictionary:
                #    option[i] = 'UNK'   
                try: 
                    option_vec = model[option[i]]
                except KeyError:
                    option[i] = 'UNK'
                prob_arr.append(test_prob[question_word_idx-1, dictionary[option[i]]])
            max_index = np.argmax(prob_arr)
            answer.append(chr(97 + max_index)) 

predict()
index = np.arange(1040)
index = index + 1 
index = index.reshape(-1, 1)
answer = np.array(answer)
answer = answer.reshape(-1, 1)
result = np.hstack([index, answer])
np.savetxt('result.csv', 
              result, 
              delimiter = ',', 
              header = 'id,answer',
              fmt = '%s',
              comments = '') 




