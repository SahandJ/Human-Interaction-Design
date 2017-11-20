import logging

logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn
import tensorflow.contrib

tf.logging.set_verbosity(tf.logging.FATAL)


class Model():
    def __init__(self, batch_size, num_batches, seq_length, hidden_cells, n_words, num_feature=4, num_layers=1):
        with tf.variable_scope('Inputs'):
            self.input_data = tf.placeholder(tf.int32, shape=(None, None), name='Input_Data')
            self.targets = tf.placeholder(tf.int32, shape=None, name='Targets', )
            self.pkeep = tf.placeholder(tf.float32, name='pkeep')
            global_step = tf.Variable(0, trainable=False)


            starter_learning_rate = 1e-3
            self.learningRate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                           num_batches, 0.96, staircase=True,
                                                           name='Decaying_Learning_rate')

        self.labels = tf.one_hot(self.targets, num_feature, on_value=1, off_value=0)

        print('Creating a Embedding Layer...')

        embedding_layer = self.embedding_layer(self.input_data,n_words,seq_length)

        print('Creating Cells...')

        rnn_layer=self.rnn_layer(embedding_layer,hidden_cells,batch_size,num_layers,self.pkeep)

        #print('Creating a Fully Connected Layer...')

        fc1=self.fully_connected(rnn_layer,2*hidden_cells,300,name='Fully_Connected_1',activation=tf.nn.relu)

        #print('Creating a Fully Connected Layer...')

        logits =self.fully_connected(fc1,300,num_feature,name="Fully_Connected_2",save=True)

        self.probs = tf.nn.softmax(logits,name='props')


        # self.loss=tf.losses.sigmoid_cross_entropy(logits=logits,multi_class_labels=self.labels)
        # self.loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=self.labels,scope='Loss')
        with tf.variable_scope('Loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
            self.loss_sum = tf.summary.scalar('Loss', self.loss)

        self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss, global_step=global_step,
                                                                             name='Adam')

        with tf.variable_scope('Accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            self.argmax = tf.argmax(self.probs, 1,name='argmax')
            self.accuracy_sum = tf.summary.scalar('Accuracy', self.accuracy)

        print('Done!')

    def embedding_layer(self,input_data,n_words,seq_length,name="Embedding"):
        with tf.device("/cpu:0"):
            with tf.variable_scope(name):
                self.embedding = tf.get_variable("embedding", [n_words + 1, seq_length])
                inputs = tf.split(tf.nn.embedding_lookup(self.embedding, input_data), seq_length, 1)
                squeezed = [tf.squeeze(input_, [1], name='Squeezed_inputs') for input_ in inputs]
                return squeezed

    def rnn_layer(self,inputs,hidden_cells,batch_size,num_layers,pkeep,name="RNN",activation=tf.nn.relu):
            with tf.variable_scope(name):
                cell1 =rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.GRUCell(hidden_cells, activation=tf.nn.relu),
                                                                 input_keep_prob=pkeep )
                                                                 for _ in range(num_layers)], state_is_tuple=False)
                cell2=rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.GRUCell(hidden_cells, activation=tf.nn.relu),
                                                                 input_keep_prob=pkeep )
                                                                 for _ in range(num_layers)], state_is_tuple=False)
                self.initial_state = cell1.zero_state(batch_size, tf.float32)
                print('Creating the Stacked RNN...')
                #outputs, encoding = tf.contrib.rnn.static_rnn(self.cell, inputs, dtype=tf.float32, scope='Cells')
                outputs ,_,_ =tf.contrib.rnn.static_bidirectional_rnn(cell1,cell2 ,inputs,dtype=tf.float32,scope="Cells")

                # print(outputs.shape)
                outputs = outputs[-1]
                if activation is not None:
                    return activation(outputs)
                else:return outputs

    def fully_connected(self,x,in_shape,out_shape,name="Fully_Connected",save=False,activation=None):

        print('Creating %s Layer...' % name)
        with tf.variable_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_shape, out_shape], stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.05, shape=[out_shape]), name='biases')
            outputs = (tf.matmul(x, weights) + biases)

            if save:
                tf.summary.histogram('Weight', weights)
                tf.summary.histogram('Bias', biases)

            if activation is not None:
                return activation(outputs)
            else:
                return outputs

