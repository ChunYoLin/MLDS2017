import gensim
import logging
import zipfile
import tensorflow as tf
import numpy as np
import os
import collections
import csv

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', \
                    level = logging.INFO)


def _read_words(fname):
    with tf.gfile.GFile(fname, "r") as f:
        return f.read().replace('\r\n', ' ') \
                .replace('?', ' ')            \
                .replace(',', ' ')            \
                .replace('"', ' ')            \
                .replace('!', ' ')            \
                .replace(':', ' ')            \
                .replace('.', ' ')            \
                .replace(';', ' ')            \
                .replace('(', ' ')            \
                .replace(')', ' ')            \
                .replace('*', ' ')            \
                .split()

def build_wordset():
    words = []; 
    article = 0
    for fname in os.listdir('./Holmes_Training_Data'):
        sentences = _read_words('./Holmes_Training_Data' + '/' + fname)        
        words = words + sentences
        article += 1
        print("read article %d" % article)
    return words

def sentence_formulation(words, num_steps):
    new_sentences = []
    max_len = 0
    index = 0
     
    for i in range(0, len(words), num_steps):
        if i + num_steps > len(words):
            index = i
            break
        vector = words[i: i + num_steps]
        new_sentences.append(vector)

    vector = words[index: len(words)]
    now_len = num_steps - len(vector)
    while now_len > 0:
        vector.append('.')
        now_len -= 1
    new_sentences.append(vector)
   
    return new_sentences, words

    
def build_test(dictionary):
    F = open('./testing_data.csv', 'r')    
    reader  = csv.reader(F)
    next(reader, None)
    for row in reader:
        option = row[2:7]
        question = row[1]
        for i in range(len(option)):
            if option[i] not in dictionary:
                dictionary[option[i]] = len(dictionary)
     
    return dictionary

def build_dataset(words, new_sentences):      
    vocabulary_size = 12000        #most common seen words 
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    #complement the deficiency of dictionary
    # [number, question, a, b, c, d, e]
    dictionary = build_test(dictionary)

    inv_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    
    for vector in new_sentences:
        for i in range(len(vector)):
            if vector[i] not in dictionary:
                vector[i] = 'UNK' 
    return new_sentences, dictionary, inv_dictionary
    
def main():
    num_steps = 30;    

    words = build_wordset() 
    new_sentences, words = sentence_formulation(words, num_steps)
    new_sentences, dictionary, inv_dictionary = build_dataset(words, new_sentences)
    #for vector in new_sentences:
    #    print(vector)
    
    model = gensim.models.Word2Vec(new_sentences)
    print model
    model.save('./word2vec_model')
    
if __name__ == "__main__":
    main()
