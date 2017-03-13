import collections
import sys
import re
import math
import random
import numpy as np
import tensorflow as tf

vocab_size = 5000
def build_dataset(filename):
    words = list()
    with open(filename, 'r') as raw_file:
        for line_idx, word in enumerate(raw_file.read().split()):
            word = re.sub('"', '', word)
            word = re.sub(',', '', word)
            word = re.sub('\?', '', word)
            word = re.sub('!', '', word)
            word = re.sub('\.', '', word)
            if word:
                words.append(word.lower())
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    del words
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset('./data/Holmes_Training_Data/04TOM10.TXT')

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen = span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size = 32, num_skips = 2, skip_window = 1)

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


graph = tf.Graph()
num_steps = 10000
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
    train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev = 0.1 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights = nce_weights,
                        biases = nce_biases,
                        labels = train_labels,
                        inputs = embed,
                        num_sampled = num_sampled,
                        num_classes = vocab_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session(graph = graph) as session:
        init.run()
        avg_loss = 0.
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
            avg_loss += loss_val
            if step % 10 == 0:
                print("avg loss at step ", step, ": ", avg_loss / 10)
                avg_loss = 0
    final_embeddings = normalized_embeddings.eval()

            

