from corpus import build_corpus
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
corpus = build_corpus('./data/Holmes_Training_Data/14WOZ10.TXT')
model = Word2Vec.load('./gensim_wvec')
pretrain_wv = np.ndarray(shape = (len(model.vocab), 100), dtype = np.float32)
word_id = {}
inv_word_id = {}
for idx, word in enumerate(model.vocab):
    word_id[idx] = word
    inv_word_id[word] = idx
    pretrain_wv[idx] = model[word]

corpus_by_id = []
for seq in corpus:
    line = []
    for word in seq:
        if word in model.vocab:
            line.append(inv_word_id[word])
    if len(line) > 0:
        corpus_by_id.append(line)

    
lstm = tf.contrib.rnn.BasicLSTMCell(10, state_is_tuple = False)
state = tf.zeros([1, lstm.state_size], tf.float32)
embed = tf.nn.embedding_lookup(pretrain_wv, corpus_by_id[: 1])
for i in range(1):
    out, state = lstm(embed[:, i, :], state)

final_state = state
sess = tf.Session()
sess.run(tf.initialize_all_variables())

