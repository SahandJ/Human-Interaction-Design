import pymysql
import numpy as np
import pandas
import pickle
class Database:
    def __init__(self,data_split=0.9,data_amount=100000,data_parent=False):
        try:
            self.db = pymysql.connect("","","","", charset='utf8')
        except:
            print("cant connect to remote database, using local database")
            self.db = pymysql.connect("localhost", "root", "root", ", charset='utf8')
        self.data_amount = data_amount
        self.data_split = data_split
        self.label_to_database =[]
        self.label_dict = {}
        self.given_topic = []
        self.data_parent=data_parent
        self.number_labels=0


        self.cursor = self.db.cursor()
        print("Querring Databse...")



    def querry_for_training(self):



        if self.data_amount is not None:

            self.cursor.execute("SELECT uia_questioncategories.idCategory,uia_categories.idParent,uia_questions.question,"
                           "uia_questions.answer\
            FROM quest.uia_questions \
            inner join quest.uia_questioncategories ON quest.uia_questions.id=quest.uia_questioncategories.idQuestion\
            inner join quest.uia_categories on quest.uia_questioncategories.idCategory=quest.uia_categories.id limit {}".format(
                self.data_amount))
        else:
            self.cursor.execute(
                "SELECT uia_questioncategories.idCategory,uia_categories.idParent,uia_questions.question,"
                "uia_questions.answer\
                 FROM quest.uia_questions \
                 inner join quest.uia_questioncategories ON quest.uia_questions.id=quest.uia_questioncategories.idQuestion\
                 inner join quest.uia_categories on quest.uia_questioncategories.idCategory=quest.uia_categories.id")

        self.data = self.cursor.fetchall()

        self.questions = []
        self.label_targets = []
        self.answers = []

        if self.data_parent:
            print("Using Topic Parents")
            for row in self.data:
                label, question, answer = row[1], row[2], row[3]
                # print(answer)
                self.questions.append([question])
                self.label_targets.append(label)
                self.answers.append([answer])
        else:
            for row in self.data:
                label, question, answer = row[0], row[2], row[3]
                # print(answer)
                self.questions.append([question])
                self.label_targets.append(label)
                self.answers.append([answer])
            #print(self.label_targets)
            #print(set(self.label_targets))
        self.db_label_voc=dict(zip(set(self.label_targets),range(len(set(self.label_targets)))))

        with open("save/label_dict.pkl","wb") as f:
            pickle.dump(self.db_label_voc,f)

        self.label_to_database=list(map(self.db_label_voc.get,self.label_targets))

    def data_prep_text(self):
        self.querry_for_training()

        numpy_question = np.array(self.questions)
        numpy_answers =np.array(self.answers)


        X = pandas.DataFrame(numpy_question)[0]
        Y = pandas.DataFrame(numpy_answers)
        seed = np.random.permutation(X.index)
        X = X.reindex(seed)
        Y = Y.reindex(seed)

        x_train = X.sample(frac=self.data_split, random_state=1)
        y_train = Y.sample(frac=self.data_split, random_state=1)

        x_test = X.drop(x_train.index)
        y_test = Y.drop(y_train.index)

        return x_train, y_train, x_test, y_test

    def data_prep_topic(self):
        self.querry_for_training()
        numpy_question = np.array(self.questions)
        numpy_targets=np.array(self.label_to_database).astype(np.int)
        #print(self.label_targets)
        #print(self.label_to_database)

        #return numpy_question,numpy_targets,numpy_answers


        self.number_labels = len(set(numpy_targets))
        X = pandas.DataFrame(numpy_question)[0]
        Y = pandas.Series(numpy_targets)
        seed = np.random.permutation(X.index)
        X = X.reindex(seed)
        Y = Y.reindex(seed)

        x_train = X.sample(frac=self.data_split, random_state=1)
        y_train = Y.sample(frac=self.data_split, random_state=1)


        x_test = X.drop(x_train.index)
        y_test = Y.drop(y_train.index)

        x_train = np.array(list(x_train))
        y_train = np.array(list(y_train))
        y_test = np.array(list(y_test))
        y_test = np.array(list(y_test))

        #np.save("save/quest_train",x_train)
        #np.save("save/quest_test", x_test)
        #np.save("save/label_train", y_train)
        #np.save("save/label_train", y_test)

        #print('test test')
        #print(len(x_train))
        #print(len(y_train))

        return x_train,y_train,x_test,y_test





       # y_train = y_train.reindex(seed)


    def data_prep_chat(self):
        self.label_parent_dict ={}
        with open("save/label_dict.pkl","rb") as f:
           self.db_label_voc=pickle.load(f)

        self.label_dict={}
        self.cursor.execute("SELECT \
                            uia_categories.id,\
                            uia_categories.idParent,uia_categories.title\
                                    FROM quest.uia_categories")
        self.number_labels= len(self.db_label_voc)




        data=self.cursor.fetchall()
        for row in data:
            label,parent,topic = row[0],row[1],row[2]
            self.label_dict[label]=topic
            self.label_parent_dict[parent]=topic
        print("first quarry done")

        self.cursor.execute("select distinct idCategory,  max(uia_questions.answer)\
                            from uia_questioncategories\
                            join uia_questions on idQuestion=uia_questions.id group by idCategory")
        data = self.cursor.fetchall()
        print("last quarry done")
        self.label_to_answer={}
        for row in data:
            self.label_to_answer[row[0]]=row[1]
        #print(self.label_to_answer)


        #return self.label_dict,self.label_to_answer

    def data_prep_metadata_age(self):

        if self.data_amount is not None:

            self.cursor.execute('select age,question \
                             from quest.uia_questions where age > 0 limit {0}'.format(self.data_amount))
        else:
            self.cursor.execute('select age,question \
                                         from quest.uia_questions where age > 0')

        self.data = self.cursor.fetchall()

        #self.gender = []
        self.age = []
        self.questions = []
        for row in self.data:
            age, questions = row[0], row[1]
            #print(questions)
            # print(answer)
            self.age.append(age)
            self.questions.append([questions])
        #print(self.age)


       # print(self.questions[:2])
        self.db_label_voc = dict(zip(set(self.age), range(len(set(self.age)))))

        with open("save/label_dict.pkl", "wb") as f:
            pickle.dump(self.db_label_voc, f)

        self.label_to_database = list(map(self.db_label_voc.get, self.age))
        numpy_question = np.array(self.questions)
        numpy_age =np.array(self.label_to_database).astype(np.int)
        self.number_labels = len(set(numpy_age))

        X = pandas.DataFrame(numpy_question)[0]
        Y = pandas.Series(numpy_age)
        seed = np.random.permutation(X.index)
        X = X.reindex(seed)
        Y = Y.reindex(seed)

        x_train = X.sample(frac=self.data_split, random_state=1)
        y_train = Y.sample(frac=self.data_split, random_state=1)

        x_test = X.drop(x_train.index)
        y_test = Y.drop(y_train.index)

        x_train = np.array(list(x_train))
        y_train = np.array(list(y_train))
        y_test = np.array(list(y_test))
        y_test = np.array(list(y_test))
        #print(y_train)


        return x_train, y_train, x_test, y_test

    def data_prep_metadata_gender(self):
        if self.data_amount is not None:

            self.cursor.execute("select gender,question \
                              from quest.uia_questions where gender <> '' limit {0}".format(self.data_amount))
        else:
            self.cursor.execute("select gender,question \
                                          from quest.uia_questions where gender <> '' ")

        self.data = self.cursor.fetchall()


        self.gender = []
        self.questions = []
        for row in self.data:
            gender, questions = row[0], row[1]
            # print(questions)
            # print(answer)
            self.gender.append(gender)
            self.questions.append([questions])
            # print(self.age)


            # print(self.questions[:2])
        self.db_label_voc = dict(zip(set(self.gender), range(len(set(self.gender)))))

        with open("save/label_dict.pkl", "wb") as f:
            pickle.dump(self.db_label_voc, f)

        self.label_to_database = list(map(self.db_label_voc.get, self.gender))
        numpy_question = np.array(self.questions)
        numpy_age = np.array(self.label_to_database).astype(np.int)
        self.number_labels = len(set(numpy_age))

        X = pandas.DataFrame(numpy_question)[0]
        Y = pandas.Series(numpy_age)
        seed = np.random.permutation(X.index)
        X = X.reindex(seed)
        Y = Y.reindex(seed)

        x_train = X.sample(frac=self.data_split, random_state=1)
        y_train = Y.sample(frac=self.data_split, random_state=1)

        x_test = X.drop(x_train.index)
        y_test = Y.drop(y_train.index)

        x_train = np.array(list(x_train))
        y_train = np.array(list(y_train))
        y_test = np.array(list(y_test))
        y_test = np.array(list(y_test))
        # print(y_train)


        return x_train, y_train, x_test, y_test

    def prob_to_answer(self,probs):

        inv_voc = {v: k for k, v in self.db_label_voc.items()}
        print(inv_voc)
        self.database_index =list(map(inv_voc.get,probs))
        print(self.database_index)


        self.db_index_topic=list(map(self.label_dict.get,self.database_index))
        #print(self.db_index_topic)
        #print(list(map(self.label_parent_dict.get,self.database_index)))

        if self.data_parent:
            self.db_index_topic=(list(map(self.label_parent_dict.get,self.database_index)))



        self.db_answer=list(map(self.label_to_answer.get,self.database_index))
        print(self.db_answer)

        #print(self.db_index_topic)
        for i,topic in enumerate(self.db_index_topic):
            #print(i, topic)
            if topic not in self.given_topic:
                self.given_topic.append(topic)
                return self.db_index_topic[i], self.db_answer[i]

    def get_number_labels(self):
        return self.number_labels
    def clear_prob_to_answer(self):
        self.given_topic.clear()



if __name__ == '__main__':
    db=Database(data_amount=1000)
    db.data_prep_chat()
    print(db.get_number_labels())

    with open("save/vocab.pkl", "rb") as f:
        vocabulary,labels = pickle.load(f)
    print(labels)
    # #print(x_train)
    # #print(y_train)
    # from tensorflow.contrib import learn
    # db.reset_batch_pointer()
    # vocab_processor = learn.preprocessing.VocabularyProcessor(10)
    #
    # x_transform_train = vocab_processor.fit_transform(x_train)
    # n_words = len(vocab_processor.vocabulary_)
    #
    # x_transform_test = vocab_processor.transform(x_test)
    # x_test = np.array(list(x_transform_test))
    # y_test = y_test
    # print(y_test)
    #
    # x_train = np.array(list(x_transform_train))
    # # y_train = np.array(list(y_train))
    # print('Tensor questions:', x_train[0])
    # print('Labels', y_train[:10])
    # print('Vocablulary:', dict(vocab_processor.vocabulary_._mapping))
    # dict = dict(vocab_processor.vocabulary_._mapping)
    # print(x_train)
    # db.make_batches(x_train,y_train,batch_size=100)
    # print(db.next_batch())


















    #x_train, y_train, x_test, y_test=db.data_prep_labels()


    #print(x_train[:])

    #vocabalery={k:v for k,v in zip(x_train.,range(len(x_train)))}





    #print(y_train)

    #inv_voc = {v: k for k, v in voc.items()}
    #print(label)

    #database_index =list(map(inv_voc.get,test))
    #print("Argmax to index for database: ",database_index)
    #print("Database category to to Topic: ",list(map(label.get,database_index)))
    #print("Database Category index to answer: ",list((map(answer.get,database_index)))[0])
    #print(answer)




