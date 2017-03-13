import tensorflow as tf
lstm_size = 10
batch_size = 32
probabilities = []
loss = 0.0
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
state = tf.zeros([batch_size, lstm.state_size])
for current_batch_of_words in words_in_dataset:
    output, state = lstm(current_batch_of_words, state)
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)
