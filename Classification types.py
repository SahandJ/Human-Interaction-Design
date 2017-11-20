import numpy as np
from database_data import Database
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from tensorflow.contrib import learn
from tqdm import tqdm


def test(classicator, name):
    print("Starting testing %s" % name)
    clf = classicator
    print("%s: Score func test data:" % name, clf.score(x_test, y_test))
    print("Done with %s!" % name)


def train(x_train, y_train, classicator):
    # print("Training %s" % name)
    clf = classicator
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    # print("%s: Prdictred mismatch" % name, x_train.shape[0], (y_train != y_pred).sum())
    # print("%s: Vanilla:" % name, (y_train == y_pred).sum() / x_train.shape[0])


def next_batch(pointer):
    # x,y = db.next_batch()
    x, y = x_batches[pointer], y_batches[pointer]
    pointer += 1
    # #x_test, y_test = self.x_test_batches[self.pointer], self.y_test_batches[self.pointer]
    # return x, y, self.x_test, self.y_test
    return x, y


if __name__ == '__main__':
    db = Database(data_amount=200000, data_parent=True)
    x_train, y_train, x_test, y_test = db.data_prep_topic()

    vocab_processor = learn.preprocessing.VocabularyProcessor(200)
    sequence = False

    #################### No Sequence ##################################
    if sequence is False:
        train_vectorizer = CountVectorizer()

        names = [
            "Decision Tree", "Random Forest", "Naive Bayes", "MultinomialNB", "Nearest Neighbors", "Linear SVM",
            "RBF SVM"]
        classifiers = [

            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GaussianNB(),
            MultinomialNB(),
            KNeighborsClassifier(),
            SVC(kernel="linear", C=0.025),
            SVC()]

        batch_size = 100

        num_batches = int(len(x_train) / batch_size)
        print(num_batches)
        x_train = x_train[:num_batches * batch_size]
        y_train = y_train[:num_batches * batch_size]

        x_batches = np.split(x_train, num_batches)
        y_batches = np.split(y_train, num_batches)

        y_train = np.array(list(y_train))
        print(y_train)
        y_test = np.array(list(y_test))

        pointer = 0
        for _ in tqdm(range(len(x_batches))):
            # print('Number of batche: %s'%_)
            # print('Number of batches lef: %s'%(len(x_batches)-_))
            x_train, y_train = next_batch(pointer)
            # print(x)
           # train_vectorizer = CountVectorizer()
            #x_train = train_vectorizer.fit_transform(x_train).toarray()
            x_transform_train = vocab_processor.fit_transform(x_train)
            x_train = np.array(list(x_transform_train))

            # print(x_train)

            from threading import Thread

            for i, x in enumerate(classifiers):
                thread = Thread(target=train, args=(x_train, y_train, x))
                thread.start()
            thread.join()

#x_test = train_vectorizer.transform(x_test).toarray()
x_transform_test = vocab_processor.transform(x_test)
x_test = np.array(list(x_transform_test))
print("----------------------------Start TEST---------------")
for i, x in enumerate(classifiers):
    test(x, names[i])



















    #################### With Sequence ##################################
    # x_transform_train = vocab_processor.fit_transform(x_train)
    # n_words = len(vocab_processor.vocabulary_)
    #
    # x_transform_test = vocab_processor.transform(x_test)
    # x_test = np.array(list(x_transform_test))
    #
    #
    # x_train = np.array(list(x_transform_train))

    # #print(x_train)
    # print(y_test)
