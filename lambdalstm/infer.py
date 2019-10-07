try:
  import unzip_requirements
except ImportError:
  pass

import os
import json
import time
import tensorflow as tf
import boto3
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']
LSA = 'new_lsa.pkl'
TFIDF = 'smaller_vectorizer.pkl'
ENCODER = 'encoder.pkl'

filelist = [ 
        f for 
        f in os.listdir(FILE_DIR) 
    ]

for f in filelist:
    os.remove(os.path.join(FILE_DIR, f))

encoder = pickle.load(open(FILE_DIR + ENCODER, 'rb'))
smaller_vectorizer = pickle.load(open(FILE_DIR + TFIDF, 'rb'))
new_lsa = pickle.load(open(FILE_DIR + LSA, 'rb'))

graph = tf.Graph()
batch_size = 500
chunk_size = 20
output_size = 14
lstm_layers = 2
lstm_size = 256
learning_rate = 0.001
test_acc = []


with graph.as_default():
    inputs_ = tf.placeholder("float", [None, chunk_size, 100])
    labels_ = tf.placeholder(tf.int32, [None, output_size])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with graph.as_default():
    with tf.name_scope("RNN_layers"):
        def lstm_cell():
            # Your basic LSTM cell
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
            # Add dropout to the cell
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs_,
                                             initial_state=initial_state)

    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], output_size, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

def max_prediction(pd_vec):
    max_pred = max(pd_vec)
    return np.array([int(n == max_pred) for n in pd_vec])

def word_2_vect(word):
    return new_lsa.transform(smaller_vectorizer.transform([word]))[0]

def vec_to_label(pdctns):
    pred_vec = max_prediction(pdctns)
    rolled = np.roll(pred_vec, 1)
    return encoder.inverse_transform(
        list(map(lambda tup: tup[0], 
                 filter(lambda tup: tup[1], enumerate(rolled)))))

def chunks(l, n=chunk_size):
    for i in range(0, len(l), n):
        new_chunk = l[i:i+n]
        if len(new_chunk) < n:
            new_chunk = ([''] * (n - len(new_chunk))) + new_chunk
        yield new_chunk    

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def result_from_batch(prediction_batch, padding_size=0):
    result_vecs = prediction_batch[:(batch_size - padding_size)]
    sm = softmax(result_vecs.sum(axis=0))
    rolled = np.roll(sm, 1)
    max_conf = rolled.max()
    label_int = next(tup[0] for tup in enumerate(rolled) if tup[1] == max_conf)
    return encoder.inverse_transform([label_int])[0], max_conf        


def classification_engine(words):
    testing_set = []
    word_arr = words.split(' ')

    for chunk in chunks(word_arr, chunk_size):
        testing_set.append([word_2_vect(wrd) for wrd in chunk])

    # Fill batch
    padding_size = batch_size - len(testing_set)
    testing_set.extend([testing_set[0] for n in range(padding_size)])

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, FILE_DIR + "final_sentiment.ckpt")
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))
        feed = {inputs_: testing_set,
                labels_: np.array([[0] * 14]),
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state, prediction_batch = sess.run([accuracy, final_state, predictions], feed_dict=feed)

    label, confidence = result_from_batch(prediction_batch, padding_size)
    return {
        "prediction": label,
        "confidence": "{:.3f}".format(confidence)
    }

def inferHandler(event, context):
    body = json.loads(event.get('body', '{"words":""}'))
    result = classification_engine(body.get('words', ''))

    response = {
        "statusCode": 200,
        "body": json.dumps(result)
    }
    return response
