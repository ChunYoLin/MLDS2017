import os
import re
import numpy as np
from corpus import build_corpus
from gensim.models import Word2Vec 

corpus = list()
for novel in os.listdir('./data/Holmes_Training_Data'):
    corpus.extend(build_corpus('./data/Holmes_Training_Data/' + novel))
model = Word2Vec(size = 100, window = 5, min_count = 5, workers = 4)
model.build_vocab(corpus)
for i in range(5):
    model.train(corpus)
model.save('gensim_wvec')
