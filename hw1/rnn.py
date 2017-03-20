import tensorflow as tf
import numpy as np
import gensim
import os
import collections
import word2vec
import csv

data_index = 0
batch_size = 20
n_inputs = 100
n_steps = 30
n_hidden_units = 256
epoch = 2

model = gensim.models.Word2Vec.load('./word2vec_model')
#print(model['boat'])

words = word2vec.build_wordset()
new_sentences, words = word2vec.sentence_formulation(words, n_steps)
new_sentences, dictionary, inv_dictionary = word2vec.build_dataset(words, new_sentences)

print(new_sentences[0])

vocabulary_size = len(dictionary)

def generate_batch(batch_size, dictionary):
    global data_index
    global n_steps
    global n_inputs
    batch = np.ndarray([batch_size, n_steps, n_inputs], dtype = np.float32)
    labels = np.ndarray([batch_size, n_steps], dtype = np.int32)
    
    for i in range(batch_size):
        for j in range(n_steps):
            word = new_sentences[data_index][j]
            batch[i, j] = model[word]
            if j != n_steps - 1: 
                word = new_sentences[data_index][j + 1]
                labels[i, j] = dictionary[word]            
            else:                                        #last word of a sentence
                if len(new_sentences) == data_index + 1: #last word of the word set
                    word = new_sentences[data_index][j]
                else:
                    word = new_sentences[data_index + 1][0]
                labels[i, j] = dictionary[word]
        data_index += 1   
    return batch, labels

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
    print('????????????????????????????????????????')
    # X ==> (20 batch * max_len steps,  100 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X = tf.nn.dropout(X, keep_prob)
    # into hidden
    # X_in = (20 batch *  n_steps, 256 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (20 batch, n_steps, 256 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    print('????????????????????', batch_size) 
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
    results = tf.matmul(outputs, weights['out']) + biases['out']
    probs = tf.nn.softmax(results)
    #prob to predict next vocab 
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [results],
            [tf.reshape(y, [-1])],
            [tf.ones([batch_size * n_steps], dtype = tf.float32)])

    cost = tf.reduce_mean(loss)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    return results,loss, probs, cost, train_op

#prob to predict next vocab 
result_, loss_, probs_, cost_, train_op_ = RNN(x, weights, biases, batch_size)

  
 








sess = tf.Session()
sess.run(tf.global_variables_initializer())

print '= Training ='
batch_size = 20
while epoch > 0:
    while len(new_sentences) > data_index + 1:
        if len(new_sentences) - data_index < batch_size:
            break
        batch_xs, batch_ys = generate_batch(batch_size, dictionary)
        train_probs, train_loss, _ = sess.run([probs_, cost_, train_op_], 
                                 feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 0.5})
        print(train_probs.shape)
        print("Epoch %d: %d Train Perplexity: %.3f" % (epoch, data_index, train_loss))
    epoch -= 1
    data_index = 0   

print '= Prediction ='
max_len = 0
batch_size = 1
F = open('./testing_data.csv', 'r')    
reader  = csv.reader(F)
next(reader, None)
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
            question = question[0: i]
            break

    if max_len < len(question):
        max_len = len(question)    
    
    now_len = n_steps - len(question) 
    while now_len > 0:
        question.append('UNK')
        now_len -= 1 
    
    #if row[0] == '2':
    #    print(question)
    #    print(len(question))
    
    embd_question = np.ndarray([1, n_steps, n_inputs], dtype = np.float32)
    print('~~~~~~~~~~!!!!~~~~~~', embd_question.shape, batch_size)
    for i in range(n_steps):
        print i
        try:
            embd_question[0, i] = model[question[i]]
        except KeyError:
            print "not found! ",  question[i]
            embd_question[0, i] = model['UNK']
    test_prob = sess.run(probs_, feed_dict = {x: embd_question, keep_prob: 1})
    print(row[0])
    print(test_prob.shape)     

index = numpy.arange(1040)
index = index + 1
index = index.reshape(-1, 1)
'''
numpy.savetxt('result.csv', 
              result, 
              delimiter = ',', 
              header = 'id, answer',
              fmt = '%d')   
'''
print max_len

    










