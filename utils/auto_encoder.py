# coding: utf-8
import tensorflow as tf
import sys
import numpy as np
import random
import json
from data_processing import Dataset

class Encoder:
    def __init__(self, batch_size, training_epochs, display_iter, model_path, continue_train,
                 init_learning_rate, decay_steps, decay_rate, min_learning_rate, threshold, alpha,
                 timesteps, feature_len, encoder_n_hiddens, decoder_n_hiddens, n_rep):
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.display_iter = display_iter
        self.model_path = model_path
        self.continue_train = continue_train
        self.init_learning_rate = init_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_learning_rate = min_learning_rate
        self.threshold = threshold
        self.alpha = alpha
        self.timesteps = timesteps
        self.feature_len = feature_len
        self.encoder_n_hiddens = encoder_n_hiddens
        self.decoder_n_hiddens = decoder_n_hiddens
        self.n_rep = n_rep
        self.hint()
        self.build_inputs()
        self.build_model()
        self.build_loss()
        self.saver = tf.train.Saver()
        self.build_optimizer()

    def hint(self):
        print "number of bits in representation:", self.n_rep
        print "number of hidden layers in encoder:", self.encoder_n_hiddens
        print "number of hidden layers in decoder:", self.decoder_n_hiddens
        print "alpha (threshold):", self.alpha
        print "threshold:", self.threshold

    def build_inputs(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.timesteps, self.feature_len))
        self.affinity = tf.placeholder(tf.float32, (None, None))

    def get_a_cell(self, n_hiddens, project_dim=None):
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(h) for h in n_hiddens])
        if project_dim:
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, project_dim)
        return cell

    def build_model(self):
        with tf.variable_scope('encoder'):
            encoded_ouput, _ = tf.nn.dynamic_rnn(cell=self.get_a_cell(self.encoder_n_hiddens, project_dim=self.n_rep),
                                                                        inputs=self.inputs,
                                                                        dtype=tf.float32
                                                                        )

        # the representation of the sequence is the last hidden state, shape: [batch_size, n_rep]
        self.rep = encoded_ouput[:, -1]
        # the input of each timestep for decoder is the representation, shape: [batch_size, n_timestep, n_rep]
        decoder_inputs = tf.tile(tf.expand_dims(self.rep, 1), multiples=[1, self.timesteps, 1])

        with tf.variable_scope('decoder'):
            #decoder_inputs = tf.zeros_like(self.inputs)
            decoded_outputs, _ = tf.nn.dynamic_rnn(cell=self.get_a_cell(self.decoder_n_hiddens, project_dim=self.feature_len),
                                                   inputs=decoder_inputs,
                                                   #initial_state=self.rep,
                                                   dtype=tf.float32
                                                   )

        self.decoded_output = tf.nn.softmax(tf.squeeze(decoded_outputs, -1))

    def build_loss(self):
        # TODO: change loss function
        # reconstruction loss from hour scope and day scope
        self.se = tf.reduce_sum(tf.square(tf.squeeze(self.inputs, -1) - self.decoded_output))
        # peer similarity loss
        repMat = tf.tile(tf.expand_dims(self.rep, 1), multiples=[1, self.batch_size, 1])
        repMat_t = tf.tile(tf.expand_dims(self.rep, 0), multiples=[self.batch_size, 1, 1])
        # activate similarity with threshold
        affinity_with_threshold = tf.where(tf.greater_equal(self.affinity, self.threshold), self.affinity, tf.zeros_like(self.affinity))
        s = tf.tile(tf.expand_dims(affinity_with_threshold, 2), multiples=[1, 1, self.n_rep])
        # tf.select((tf.sigmoid(hid_state) > 0.5)
        self.peer_loss = tf.reduce_sum(tf.multiply(tf.square(repMat - repMat_t), s))

        self.loss = self.se + self.alpha * self.peer_loss

    def build_optimizer(self):
        self.current_epoch = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(self.init_learning_rate,
                                                        self.current_epoch,
                                                        decay_steps=self.decay_steps,
                                                        decay_rate=self.decay_rate)
        self.learning_rate = tf.maximum(self.learning_rate, self.min_learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.current_epoch)

    def train(self, data, affinity):
        batch_start = 0
        total_data = len(data)
        print 'Start training'
        with tf.Session() as sess:
            if self.continue_train:
                try:
                    self.saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                    print 'Successfully load model!'
                except:
                    print 'Fail to load model!'
                    sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())

            step = 1
            epoch_loss = []
            while step <= self.training_epochs:
                self.current_epoch = step
                batch_range = np.arange(batch_start, min(total_data, batch_start + self.batch_size))
                if total_data - batch_start < self.batch_size:
                    rand_id = np.random.randint(low=0, high=total_data, size=(self.batch_size - (total_data - batch_start)))
                    batch_range = np.concatenate((batch_range, rand_id))
                batch_x = data[batch_range]
                batch_affinity = affinity[batch_range][:, batch_range]

                feed = {
                    self.inputs: batch_x,
                    self.affinity: batch_affinity
                }
                _, loss, se, pl, lr, decoded = sess.run([self.optimizer, self.loss, self.se, self.peer_loss, self.learning_rate, self.decoded_output], feed_dict=feed)
                epoch_loss.append(loss)

                if total_data - batch_start > self.batch_size:
                    batch_start += self.batch_size
                else:
                    # last batch
                    # print the result of last epoch
                    print batch_x[0].reshape((-1))
                    print decoded[0]
                    print se, self.alpha * pl
                    print "Epoch " + str(self.current_epoch) + ", Learning Rate = " + "{:.5f}".format(lr) + ", Average Loss = " + "{:.7f}".format(np.mean(epoch_loss))
                    # set the start index to 0
                    batch_start = 0
                    step += 1
                    epoch_loss = []
                    # shuffle data for next epoch
                    zipped_data = zip(data, affinity)
                    random.shuffle(zipped_data)
                    data[:], affinity[:] = zip(*zipped_data)

            print("Optimization Finished!")
            self.saver.save(sess, self.model_path+'model')

    def get_representations(self, data, affinity):

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            feed = {
                self.inputs: data,
                self.affinity: affinity
            }

            rep = sess.run(self.rep, feed_dict=feed)

            return rep

if __name__ == '__main__':
    seq = sys.argv[1]
    dataset = Dataset(data_rpath='../data/collection.json',
                      weekly_distance_rpath='../data/weekly_distance',
                      daily_distance_rpath='../data/daily_distance'
                      )

    data = dataset.data[seq].reshape(-1, 24, 1)
    affinity = np.exp(-dataset.distance[seq] / dataset.distance[seq].std())
    encoder = Encoder(batch_size=256,
                      training_epochs=500,
                      display_iter=10,
                      model_path=seq+'_model/',
                      init_learning_rate=0.01,
                      decay_steps=1000,
                      decay_rate=0.9,
                      min_learning_rate=0.00001,
                      encoder_n_hiddens=[32, 16, 4],
                      decoder_n_hiddens=[32, 16, 4],
                      n_rep=2,
                      threshold=np.percentile(affinity, 80),
                      alpha=0.01,
                      continue_train=False,
                      timesteps=24,
                      feature_len=1
                      )
    encoder.train(data, affinity)
    rep = encoder.get_representations(data, affinity)

    with open('../data/' + seq + '_encoded.json', 'w') as wf:
        json.dump(rep.tolist(), wf)

