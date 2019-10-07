import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

class Predictor:
    class __Predictor:
        def __init__(self):
            print('Loading encoder...')
            self.encoder = pickle.load(open('./pickles/encoder.pkl', 'rb'))
            print('Loading vectorizer...')
            self.smaller_vectorizer = pickle.load(open('./pickles/smaller_vectorizer.pkl', 'rb'))
            print('Loading lsa...')
            self.new_lsa = pickle.load(open('./pickles/new_lsa.pkl', 'rb'))

            self.graph = tf.Graph()
            self.batch_size = 500
            self.chunk_size = 20
            self.output_size = 14
            self.lstm_layers = 2
            self.lstm_size = 256
            self.learning_rate = 0.001
            self.test_acc = []

            with self.graph.as_default():
                self.inputs_ = tf.placeholder("float", [None, self.chunk_size, 100])
                self.labels_ = tf.placeholder(tf.int32, [None, self.output_size])
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

                with self.graph.as_default():
                    with tf.name_scope("RNN_layers"):
                        def lstm_cell():
                            # Your basic LSTM cell
                            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, reuse=tf.get_variable_scope().reuse)
                            # Add dropout to the cell
                            return tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=self.keep_prob)

                # Stack up multiple LSTM layers, for deep learning
                self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.lstm_layers)])

                # Getting an initial state of all zeros
                self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

                self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.inputs_,
                                                        initial_state=self.initial_state)

                self.predictions = tf.contrib.layers.fully_connected(self.outputs[:, -1], self.output_size, activation_fn=tf.sigmoid)
                self.cost = tf.losses.mean_squared_error(self.labels_, self.predictions)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

                self.correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), self.labels_)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

                self.saver = tf.train.Saver()

        def max_prediction(self, pd_vec):
            max_pred = max(pd_vec)
            return np.array([int(n == max_pred) for n in pd_vec])

        def word_2_vect(self, word):
            return self.new_lsa.transform(self.smaller_vectorizer.transform([word]))[0]

        def vec_to_label(self, pdctns):
            pred_vec = self.max_prediction(pdctns)
            rolled = np.roll(pred_vec, 1)
            return self.encoder.inverse_transform(
                list(map(lambda tup: tup[0],
                        filter(lambda tup: tup[1], enumerate(rolled)))))

        def chunks(self, l, n=20):
            for i in range(0, len(l), n):
                new_chunk = l[i:i+n]
                if len(new_chunk) < n:
                    new_chunk = ([''] * (n - len(new_chunk))) + new_chunk
                yield new_chunk

        def softmax(self, x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        def result_from_batch(self, prediction_batch, padding_size=0):
            result_vecs = prediction_batch[:(self.batch_size - padding_size)]
            sm = self.softmax(result_vecs.sum(axis=0))
            rolled = np.roll(sm, 1)
            max_conf = rolled.max()
            label_int = next(tup[0] for tup in enumerate(rolled) if tup[1] == max_conf)
            return self.encoder.inverse_transform([label_int])[0], max_conf


        def classification_engine(self, words):
            testing_set = []
            if type(words) == list:
                word_arr = words
            elif type(words) == str:
                word_arr = words.split(' ')

            print(word_arr)
            for chunk in self.chunks(word_arr, self.chunk_size):
                testing_set.append([self.word_2_vect(wrd) for wrd in chunk])

            # Fill batch
            padding_size = self.batch_size - len(testing_set)
            testing_set.extend([testing_set[0] for n in range(padding_size)])

            print(len(testing_set))
            with tf.Session(graph=self.graph) as sess:
                self.saver.restore(sess, "pickles/final_sentiment.ckpt")
                test_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
                feed = {self.inputs_: testing_set,
                        self.labels_: np.array([[0] * 14]),
                        self.keep_prob: 1,
                        self.initial_state: test_state}
                batch_acc, test_state, prediction_batch = sess.run([self.accuracy, self.final_state, self.predictions], feed_dict=feed)

            label, confidence = self.result_from_batch(prediction_batch, padding_size)
            return label, "{:.3f}".format(confidence)


    def __getattr__(self, name):
        return getattr(self.instance, name)

    instance = None

    def __init__(self):
        if not Predictor.instance:
            print('there was not an instance')
            Predictor.instance = Predictor.__Predictor()
        else:
            print('there was an instance')

    def predict(self, words):
        result = self.instance.classification_engine(words)
        return result
