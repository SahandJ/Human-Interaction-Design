import argparse
import pickle
import re
from heapq import nlargest

import numpy as np
import tensorflow as tf

from database_data import Database
from model import Model

parser = argparse.ArgumentParser()  
group = parser.add_mutually_exclusive_group()
group.add_argument("--train",default=True,help="Start training the network on parameters given",action="store_true")
group.add_argument("--chat",default=False,help="Start the model in interactive chatting mode",action="store_true")
parser.add_argument("--data_limit",default=10000,help="Choose the data limit for database")
parser.add_argument('--batch_size', type=int, default=100, help="The size of the batch size")
parser.add_argument('--seq_length', type=int, default=256, help="The size of the the sequence of the model")
parser.add_argument('--hidden_cells', type=int, default=256, help="The number of hidden cell in the RNN network")
parser.add_argument('--number_layers', type=int, default=2, help="The number of layers in the RNN network")
parser.add_argument('-e','--num_epoch', type=int, default=5, help="The number of hidden cell in the RNN network")
parser.add_argument('--topic_parent',default=False,help="Choose to classify with the topic parents",action="store_true")
parser.add_argument('-c','--classification',default="topic", type=str, help="What kind of classification the model should train on, topic, age or gender")
parser.add_argument('--verbose', help="Show extra debug options", action="store_true")
parser.add_argument('-s','--save', default=True,help="Save checkpoint underway of training to be used later",action="store_true")
#parser.add_argument('--interactive', help="Start the chat in Interactive mode", action="store_true")


args = parser.parse_args()

class AI():
    def __init__(self,chating=False):
        tf.reset_default_graph()
        self.seq_length = args.seq_length
        print("Starting up AI...")
        if args.chat or chating:

            with open("save/vocab.pkl", "rb") as f:
                self.vocabulary, self.number_labels= pickle.load(f)
            n_words = len(self.vocabulary)
            print("   Gathering Data from Database...\n")
            self.db = Database(data_parent=args.topic_parent)
            self.db.data_prep_chat()
            print(self.number_labels)
            #self.number_labels=self.db.get_number_labels()
            print(self.number_labels)

            print("Setting up AI for Chatting...")
            self.model = Model(batch_size=1, num_batches=1,
                                seq_length=args.seq_length, 
                                hidden_cells=args.hidden_cells,
                                n_words=n_words, num_feature=self.number_labels, num_layers=args.number_layers)
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            self.ckpt = tf.train.latest_checkpoint('save/topic')
            ckpt = tf.train.latest_checkpoint('save/topic')
            print(ckpt)
            self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt),
                clear_devices=True)
            # print(self.ckpt)
            # self.saver = tf.train.Saver(tf.global_variables())
            # self.ckpt = tf.train.latest_checkpoint('save/')
            # print(self.ckpt)
            self.saver.restore(self.sess, self.ckpt,)
            print("Setup Complete.")

            # graph = tf.get_default_graph()
            # input_data = graph.get_tensor_by_name("Input_Data:0")
            # targets = graph.get_tensor_by_name("Targets:0")
            # pkeep = graph.get_tensor_by_name("pkeep:0")



        elif args.train:
            from train import main
            main(batch_size=args.batch_size,num_epoch=args.num_epoch,hidden_cells=args.hidden_cells,
                 seq_length=args.seq_length,number_layers=args.number_layers,classification=args.classification,data_amount=args.data_limit,save=args.save,data_parent=args.topic_parent)

    def get_vocabulary(self):
        return self.vocabulary

    def ask_ai(self,padded_array):
        feed = {self.model.input_data: padded_array, self.model.pkeep: 1.0}
        prob, argmax = self.sess.run([self.model.probs, self.model.argmax], feed)
        prob = prob[0]
        # print(prob)
        top_topic = nlargest(5, range(len(prob)), key=lambda x: prob[x])
        # print(top_topic)
        topic, answer = (self.db.prob_to_answer(top_topic))

        print("Jeg tror du snakker om %s" % topic)
        # print("Vi har snakket om ",)
        print(answer)

        return topic,answer

    def ask_ai_v2(self,input_to_list):



        input_to_list = (list(map(self.vocabulary.get, input_to_list)))

        filtered_list = list(filter(lambda x: x is not None, input_to_list))

        padded_array = np.array([filtered_list + [0] * (args.seq_length - len(filtered_list))])

        print(padded_array)

        feed = {self.model.input_data: padded_array, self.model.pkeep: 1.0}
        prob, argmax = self.sess.run([self.model.probs, self.model.argmax], feed)
        prob = prob[0]
        # print(prob)
        top_topic = nlargest(5, range(len(prob)), key=lambda x: prob[x])
        print("top:::",top_topic)
        try:
            topic, answer = (self.db.prob_to_answer(top_topic))
        except ValueError:
            self.db.clear_prob_to_answer()
            topic, answer = (self.db.prob_to_answer(top_topic))


        print("Jeg tror du snakker om %s" % topic)
        # print("Vi har snakket om ",)
        print(answer)
        topic_answer =[]




        return topic,answer






class Chat():

    def __init__(self):
        print("Chatting Functions Activates....")
        print("\n"*100)
        print("Hei, Hva lurer du på?")
        while True:


            chat_input = input()
            # input_to_list = chat_input.split(" ")
            chat_input = chat_input.replace("å", 'a')
            chat_input = chat_input.replace("ø", 'o')
            chat_input = chat_input.replace("æ", 'ae')
            input_to_list = re.sub('[^a-zA-Z0-9æøåØÆÅ ]', '', chat_input)
            input_to_list = input_to_list.split(" ")
            print(input_to_list)

            input_to_list = (list(map(ai.get_vocabulary().get, input_to_list)))

            filtered_list = list(filter(lambda x: x is not None, input_to_list))

            padded_array = np.array([filtered_list + [0] * (args.seq_length - len(filtered_list))])
            # padded_array = np.array(padded_array)
            print(padded_array)
            topic, answer = ai.ask_ai(padded_array)
            print("Jeg tror du snakker om %s" % topic)
            # print("Vi har snakket om ",)
            print(answer)


if __name__ == '__main__':
    ai = AI(chating=False)
    #chat = Chat()

    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    # ckpt = tf.train.latest_checkpoint('save/')
    # print(ckpt)
    # new_saver = tf.train.import_meta_graph('{}.meta'.format(ckpt),
    #   clear_devices=True)
    # new_saver.restore(sess, ckpt)
    # hparams = tf.get_collection("hparams")
    # print(hparams)