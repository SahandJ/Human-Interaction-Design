import os
import time
from sys import exit
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from database_data import Database


class data_loader:
    def __init__(self, batch_size, seq_length,classifcation,data_amount =None,data_parent=False):
        self.batch_size=batch_size
        self.num_batches = 0
        self.seq_length = seq_length
        self.number_labels = 0
        self.data_amount=data_amount
        self.classifcation=classifcation
        self.data_parent =data_parent



        self.reset_batch_pointer()
        self.prosess()

    def prosess(self):


        if self.data_amount is not None:
            db = Database(data_amount=self.data_amount,data_parent=self.data_parent)
        else:
            db=Database(None)

        if self.classifcation =="topic":
            x_train, y_train, x_test, y_test = db.data_prep_topic()
        elif self.classifcation=="age":
            x_train, y_train, x_test, y_test = db.data_prep_metadata_age()
        elif self.classifcation =="gender":
            x_train, y_train, x_test, y_test = db.data_prep_metadata_gender()

        #print((x_train))
        #print((y_train))

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.seq_length)

        x_transform_train = self.vocab_processor.fit_transform(x_train)
        n_words = len(self.vocab_processor.vocabulary_)

        x_transform_test = self.vocab_processor.transform(x_test)
        self.x_test = np.array(list(x_transform_test))
        self.y_test = y_test
        #print(self.y_test)

        x_train = np.array(list(x_transform_train))
        # y_train = np.array(list(y_train))
        print('Tensor questions:', x_train[0])
        print('Labels', y_train[:10])
        print('Vocablulary:', dict(self.vocab_processor.vocabulary_._mapping))
        self.dict = dict(self.vocab_processor.vocabulary_._mapping)

        print('Total words: %d' % n_words)

        with open("save/vocab.pkl", "wb") as f:
            pickle.dump((self.dict,len(set(y_train))), f)

        # print(len(x_train))
        self.number_labels = len(set(y_train))
        # self.number_labels =int(max(y_train))

        #db.make_batches(x_train,y_train,)



        self.num_batches = int(len(x_train) / self.batch_size)
        #print(self.num_batches)
        x_train = x_train[:self.num_batches * self.batch_size]
        y_train = y_train[:self.num_batches * self.batch_size]

        self.x_batches = np.split(x_train, self.num_batches)
        self.y_batches = np.split(y_train, self.num_batches)


    def next_batch(self):
        #x,y = db.next_batch()
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        # #x_test, y_test = self.x_test_batches[self.pointer], self.y_test_batches[self.pointer]
        # return x, y, self.x_test, self.y_test
        return x,y,self.x_test,self.y_test

    def reset_batch_pointer(self):
        self.pointer = 0

from model import Model

def main(batch_size,num_epoch,hidden_cells,seq_length,number_layers,save,classification,data_amount,data_parent):

    number_layers = number_layers
    seq_length = seq_length
    hidden_cells = hidden_cells
    num_epoch = num_epoch
    batch_size = batch_size
    data_parent = data_parent

    dataloader = data_loader(batch_size=batch_size, seq_length=seq_length, classifcation=classification,data_amount=data_amount,data_parent=data_parent)
    num_batches =dataloader.num_batches
    n_words = len(dataloader.vocab_processor.vocabulary_)
    print('label', dataloader.number_labels)
    number_labels = dataloader.number_labels

    print('Batch Size:', batch_size)
    print('Number of Batches:', num_batches)

    print('Creating the model...')
    model = Model(batch_size=batch_size, num_batches=num_batches, seq_length=seq_length, hidden_cells=hidden_cells,
                  n_words=n_words,
                  num_feature=number_labels,
                  num_layers=number_layers)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.ERROR)
        merge = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter('C:\\ftb\\test\\{}batch-{}hidden-{}labels-training'.format(batch_size, hidden_cells,number_labels),
                                            graph=sess.graph)
        writer = tf.summary.FileWriter('C:\\ftb\\test\\{}batch-{}hidden-{}labels-testing'.format(batch_size, hidden_cells,number_labels))
        sess.run(tf.global_variables_initializer())
        state = sess.run(model.initial_state)

        for epoch in range(num_epoch):
            dataloader.reset_batch_pointer()
            for b in range(num_batches):
                batch_time = time.time()
                x_train, y_train, x_test, y_test = dataloader.next_batch()
                # print(x_train[0])
                feed = {model.input_data: x_train, model.targets: y_train, model.pkeep: 0.7}
                testfeed = {model.input_data: x_test, model.targets: y_test, model.pkeep: 1}

                _, result, losses, predict, learningrate, accuracy, test, labels, pkeep = sess.run(
                    [model.train_step, merge, model.loss, model.argmax, model.learningRate, model.accuracy, model.probs,
                     model.labels, model.pkeep], feed)
                # print((sess.run(tf.round(test))[0]))
                # print((labels[0]))
                # print(sess.run(tf.equal(sess.run(tf.round(test)),labels)))
                print(accuracy)
                # print(pkeep)
                print("{}/{} (epoch {}), train_loss = {:.6f}, time/batch ={:.3f} " \
                      .format(epoch * num_batches + b+1,
                              num_epoch * num_batches,
                              epoch + 1, losses, time.time() - batch_time))

                if (epoch * num_batches + b+1) % 100 == 0:
                    print('\nTraining...')
                    train_time = time.time()

                    print('     LearningRate:', learningrate)
                    print('     Loss:', losses)

                    print('     Accuracy:', accuracy)

                    print('     Predicted by Network:', predict)
                    print('     The True Labels:', y_train)
                    print("     Training: %s seconds" % (time.time() - train_time))
                    print()
                    test_writer.add_summary(result, (epoch * num_batches + b))

                    print('Testing...')
                    test_time = time.time()
                    test_accuracy, test_predict, losses, loss_sum, accuracy_sum = sess.run(
                        [model.accuracy, model.argmax, model.loss, model.loss_sum, model.accuracy_sum], testfeed)
                    writer.add_summary(loss_sum, (epoch * num_batches + b))
                    writer.add_summary(accuracy_sum, (epoch * num_batches + b))
                    print('     LearningRate:', learningrate)
                    print('     Loss:', losses)

                    print('     Accuracy:', test_accuracy)

                    print('     Predicted by Network:', test_predict[:10])
                    print('     The True Labels:', y_test[:10])
                    print("     Testing: %s seconds" % (time.time() - test_time))
                    print()
                    if save:
                        checkpoint_path = os.path.join('save/', 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=(epoch * num_batches + b), )
                        print("model saved to {}".format('save/model.ckpt'))

        accuracy, predict, losses, loss_sum, accuracy_sum = sess.run(
            [model.accuracy, model.argmax, model.loss, model.loss_sum, model.accuracy_sum], testfeed)

        print('     LearningRate:', learningrate)
        print('     Loss:', losses)

        print('     Accuracy:', accuracy)

        print('     Predicted by Network:', predict)
        print('     The True Labels:', y_test)
        print("     Testing: %s seconds" % (time.time() - test_time))
        print()


if __name__ == '__main__':


    number_layers = 1
    seq_length = 10
    hidden_cells = 10
    num_epoch = 10
    batch_size = 100

    dataloader = data_loader(batch_size=batch_size, seq_length=seq_length, classifcation="topic",data_amount=10000,data_parent=True)
    num_batches =dataloader.num_batches
    n_words = len(dataloader.vocab_processor.vocabulary_)
    print('label', dataloader.number_labels)
    number_labels = dataloader.number_labels

    print('Batch Size:', batch_size)
    print('Number of Batches:', num_batches)

    print('Creating the model...')
    model = Model(batch_size=batch_size, num_batches=num_batches, seq_length=seq_length, hidden_cells=hidden_cells,
                  n_words=n_words,
                  num_feature=number_labels,
                  num_layers=number_layers)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.ERROR)
        merge = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter('C:\\ftb\\test\\{}batch_{}hidden'.format(num_batches, hidden_cells),
                                            graph=sess.graph)
        writer = tf.summary.FileWriter('C:\\ftb\\test\\{}batch_{}hidden_testing'.format(num_batches, hidden_cells))
        sess.run(tf.global_variables_initializer())
        state = sess.run(model.initial_state)

        for epoch in range(num_epoch):
            dataloader.reset_batch_pointer()
            for b in range(num_batches):
                batch_time = time.time()
                x_train, y_train, x_test, y_test = dataloader.next_batch()
                # print(x_train[0])
                feed = {model.input_data: x_train, model.targets: y_train, model.pkeep: 0.7}
                testfeed = {model.input_data: x_test, model.targets: y_test, model.pkeep: 1}

                _, result, losses, predict, learningrate, accuracy, test, labels, pkeep = sess.run(
                    [model.train_step, merge, model.loss, model.argmax, model.learningRate, model.accuracy, model.probs,
                     model.labels, model.pkeep], feed)
                # print((sess.run(tf.round(test))[0]))
                # print((labels[0]))
                # print(sess.run(tf.equal(sess.run(tf.round(test)),labels)))
                print(accuracy)
                # print(pkeep)
                print("{}/{} (epoch {}), train_loss = {:.6f}, time/batch ={:.3f} " \
                      .format(epoch * num_batches + b,
                              num_epoch * num_batches,
                              epoch + 1, losses, time.time() - batch_time))

                if (epoch * num_batches + b) % 100 == 0:
                    print('\nTraining...')
                    train_time = time.time()

                    print('     LearningRate:', learningrate)
                    print('     Loss:', losses)

                    print('     Accuracy:', accuracy)

                    print('     Predicted by Network:', predict[:10])
                    print('     The True Labels:', y_train[:10])
                    print("     Training: %s seconds" % (time.time() - train_time))
                    print()
                    test_writer.add_summary(result, (epoch * num_batches + b))

                    print('Testing...')
                    test_time = time.time()
                    test_accuracy, test_predict, losses, loss_sum, accuracy_sum = sess.run(
                        [model.accuracy, model.argmax, model.loss, model.loss_sum, model.accuracy_sum], testfeed)
                    writer.add_summary(loss_sum, (epoch * num_batches + b))
                    writer.add_summary(accuracy_sum, (epoch * num_batches + b))
                    print('     LearningRate:', learningrate)
                    print('     Loss:', losses)

                    print('     Accuracy:', test_accuracy)

                    print('     Predicted by Network:', test_predict[:10])
                    print('     The True Labels:', y_test[:10])
                    print("     Testing: %s seconds" % (time.time() - test_time))
                    print()

                    checkpoint_path = os.path.join('save/', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=(epoch * num_batches + b), )
                    print("model saved to {}".format('save/model.ckpt'))

        accuracy, predict, losses, loss_sum, accuracy_sum = sess.run(
            [model.accuracy, model.argmax, model.loss, model.loss_sum, model.accuracy_sum], testfeed)

        print('     LearningRate:', learningrate)
        print('     Loss:', losses)

        print('     Accuracy:', accuracy)

        print('     Predicted by Network:', predict)
        print('     The True Labels:', y_test)
        print("     Testing: %s seconds" % (time.time() - test_time))
        print()

