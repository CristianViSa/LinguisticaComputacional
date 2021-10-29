import itertools
import os

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from pandas.io.formats import string
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import csr_matrix
import re
import string
import scipy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import plot_confusion_matrix

DATA_PATH = "data/"
TEST_DATA_PATH = "TASS2017_T1_test.xml"
DEV_DATA_PATH = "TASS2017_T1_development.xml"
TRAIN_DATA_PATH = "TASS2017_T1_training.xml"
LEXICON_DATA_PATH = "ElhPolar_esV1.lex"
NEGATIVE_TAG = 'negative'
POSITIVE_TAG = 'positive'
BEST_MODEL = "SVC"
OUTPUT_FILENAME = "CristianVillarroya_"

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# def plot_confusion_matrix(confmat, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.imshow(confmat, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     thresh = confmat.max() / 2.
#     for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
#         plt.text(j, i, confmat[i, j],
#                  horizontalalignment="center",
#                  color="white" if confmat[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()


def tokenize_tweet(tweets):
    tweet_tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False)
    tweet_tokens = []

    for tweet in tweets:
        tweet_tokens.append(' '.join(tweet_tokenizer.tokenize(tweet)))
    return tweet_tokens


def read_xml(filename, newfilename=""):
    if os.path.isfile(filename):
        file = open(filename, encoding="utf8").read()
        xml = BeautifulSoup(file, 'lxml')
        tweets = [twt.text for twt in xml.findAll('content')]
        labels = [labs.text for labs in xml.findAll('value')]
        ids = [ids.text for ids in xml.findAll('tweetid')]
        df = pd.DataFrame()
        df["ids"] = ids
        df["tweets"] = tweets
        df["labels"] = labels
        if newfilename != "":
            df.to_csv("data_cleaned_" + newfilename + ".csv", sep=";", index=False)
        return df
    print("Error, no such file")
    return None


def remove_stop_words(tweet):
    stop_words = stopwords.words("spanish")
    tweet = tweet.split()
    tweet = [x.lower() for x in tweet if x.lower() not in stop_words]
    tweet = ' '.join(tweet)
    return tweet


def lemmatize_tweet(tweet):
    stemmer = PorterStemmer()
    return stemmer.stem(tweet)


# Ojo, quita horas y demas
def remove_punctuation_signs(tweet):
    text = [char for char in tweet if char not in string.punctuation]
    cleaned_text = ''.join(text)
    return cleaned_text


def polarize_tweet(tweet):
    lex_file = open(DATA_PATH + LEXICON_DATA_PATH, 'r')
    lex_dict = {}
    for line in lex_file:
        if '#' not in line:
            line = re.sub('\n', '', line)
            info = line.split('\t')
            if len(info) > 1:
                text = info[0]
                polarity = info[1]
                lex_dict[text] = polarity

    positives = 0
    negatives = 0
    for word in tweet.split():
        # print(word)
        if word in lex_dict.keys():
            if lex_dict[word] == NEGATIVE_TAG:
                negatives += 1
            else:
                positives += 1
    return positives, negatives


def preprocess_data(data):
    data = tokenize_tweet(data)
    cleaned_tweets = []
    for tweet in data:
        tweet = tweet.lower()
        # tweet = re.sub('[\w\s]', 'emoticon', tweet)
        tweet = re.sub('[^0-9a-z #@]', '', tweet)
        tweet = re.sub('[\n]', ' ', tweet)
        tweet = re.sub('@[0-9a-z]*', 'usuario', tweet)
        tweet = re.sub('#[0-9a-z]*', 'hashtag', tweet)
        tweet = re.sub('(http|https)\:\/\/[a-zA-Z0-9\.\/\?\:@\-_=#]+\.[a-zA-Z]{2,6}[a-zA-Z0-9\.\&\/\?\:@\-_=#~%]*',
                       '', tweet)
        tweet = re.sub('\d+([\.\,]\d+)?', 'numero', tweet)
        tweet = re.sub('[A-Za-z\.]*\@[a-zA-Z]*(\.[a-zA-Z]{1,6})+', 'correo', tweet)
        # Stopwords
        tweet = remove_stop_words(tweet)
        # Punctuation Signs
        tweet = remove_punctuation_signs(tweet)
        # Lemmatize
        tweet = lemmatize_tweet(tweet)
        cleaned_tweets.append(tweet)
    return cleaned_tweets

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dev = read_xml(DATA_PATH + DEV_DATA_PATH, "dev")
    test = read_xml(DATA_PATH + TEST_DATA_PATH, "test")
    train = read_xml(DATA_PATH + TRAIN_DATA_PATH, "train")
    # preprocess_data(dev['tweets'])
    dev['tweets'] = preprocess_data(dev['tweets'])
    test['tweets'] = preprocess_data(test['tweets'])
    train['tweets'] = preprocess_data(train['tweets'])
    train_positives = []
    train_negatives = []
    dev_positives = []
    dev_negatives = []
    test_positives = []
    test_negatives = []
    for tweet in train['tweets']:
        positive, negative = polarize_tweet(tweet)
        train_positives.append(positive)
        train_negatives.append(negative)
    for tweet in dev['tweets']:
        positive, negative = polarize_tweet(tweet)
        dev_positives.append(positive)
        dev_negatives.append(negative)
    for tweet in test['tweets']:
        positive, negative = polarize_tweet(tweet)
        test_positives.append(positive)
        test_negatives.append(negative)

    # vectorizer = CountVectorizer(
    #     tokenizer=TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False).tokenize)
    # vectorizer = HashingVectorizer(tokenizer=TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False).tokenize)
    # vectorizer = TfidfVectorizer()
    # vectorizer = TfidfVectorizer(min_df=8, ngram_range=(2,4), use_idf=True, smooth_idf=True, sublinear_tf=True)
    vectorizer = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])

    train_vectors = vectorizer.fit_transform(train['tweets'])
    dev_vectors = vectorizer.transform(dev['tweets'])
    test_vectors = vectorizer.transform(test['tweets'])

    train_polarity = np.transpose(np.array([train_positives, train_negatives]))
    test_polarity = np.transpose(np.array([test_positives, test_negatives]))
    dev_polarity = np.transpose(np.array([dev_positives, dev_negatives]))
    # train_polarity = scaler.fit_transform(train_polarity)
    # test_polarity = scaler.transform(test_polarity)
    # dev_polarity = scaler.transform(dev_polarity)

    # train_M = scipy.sparse.hstack((train_vectors, train_polarity))
    # test_M = scipy.sparse.hstack((test_vectors, test_polarity))
    # dev_M = scipy.sparse.hstack((dev_vectors, dev_polarity))

    train_M = train_vectors
    test_M = test_vectors
    dev_M = dev_vectors
    # # # PROBAR TAMBIEN SVM, NO LINEAR SVC
    # #
    classifier_liblinear = svm.LinearSVC(C=1.2)
    classifier_liblinear.fit(train_M, train['labels'])
    prediction_liblinear = classifier_liblinear.predict(dev_M)
    # print(prediction_liblinear)
    print(classification_report(dev['labels'], prediction_liblinear))
    print("accuracy= ", accuracy_score(dev['labels'], prediction_liblinear))
    print("macro= ", precision_recall_fscore_support(dev['labels'], prediction_liblinear, average='macro'))
    print("micro= ", precision_recall_fscore_support(dev['labels'], prediction_liblinear, average='micro'))
    confm = confusion_matrix(dev['labels'], prediction_liblinear)
    # plot_confusion_matrix(confm, classes=['N', 'P', 'NEU', 'NONE'])
    confusion_matrix(classifier_liblinear, dev_M['Tweets'], dev['labels'])
    # plt.show()
    # plot_confusion_matrix(classifier_liblinear, dev_M, dev['labels'])
    # plt.show()

    # SVC
    # Sin lematizar
    # Precision 0.54
    # macro=  (0.3789429246876056, 0.3660467749571287, 0.344770159888773, None)
    # micro=  (0.5355731225296443, 0.5355731225296443, 0.5355731225296443, None)
    # Lematizando
    # Precision 0.53
    # macro = (0.34585647554758037, 0.3547251886905386, 0.32902690590984607, None)
    # micro = (0.5276679841897233, 0.5276679841897233, 0.5276679841897233, None)
    #
    # clf = LogisticRegression(random_state=0).fit(train_M, train['labels'])
    # prediction_logistic = clf.predict(dev_M)
    # print(classification_report(dev['labels'], prediction_logistic))
    # print("accuracy= ",accuracy_score(dev['labels'], prediction_logistic))
    # print("macro= ", precision_recall_fscore_support(dev['labels'],  prediction_logistic, average='macro'))
    # print("micro= ", precision_recall_fscore_support(dev['labels'],  prediction_logistic, average='micro'))

    # Logistic Regression
    # Sin lematizar
    # Precision 0.53
    # macro=  (0.38514302664021804, 0.3655428119755004, 0.3473947155670756, None)
    # micro=  (0.5335968379446641, 0.5335968379446641, 0.5335968379446641, None)
    # Lematizando
    # Precision 0.53
    # macro = (0.35721534595468496, 0.35608627053107256, 0.33004079123264785, None)
    # micro = (0.5316205533596838, 0.5316205533596838, 0.5316205533596838, None)
    #
    # accuracy = 0.5316205533596838
    # macro = (0.3417533469616803, 0.35821475054817675, 0.3286960506167526, None)
    # micro = (0.5316205533596838, 0.5316205533596838, 0.5316205533596838, None)
    #
    # rf = RandomForestClassifier(n_estimators=100, random_state=0)
    # rf.fit(train_M, train['labels'])
    # prediction_rf = rf.predict(dev_M)
    # #print(prediction_liblinear)
    # print(classification_report(dev['labels'], prediction_rf))
    # print("accuracy= ",accuracy_score(dev['labels'], prediction_rf))
    # print("macro= ", precision_recall_fscore_support(dev['labels'],  prediction_rf, average='macro'))
    # print("micro= ", precision_recall_fscore_support(dev['labels'],  prediction_rf, average='micro'))
    # #
    #
    # # Escribir resultado en fichero de salida
    # output = open(OUTPUT_FILENAME + BEST_MODEL + ".txt", "w")
    # prediction_liblinear = classifier_liblinear.predict(test_M)
    # for index in range(len(prediction_liblinear)):
    #     id = test['ids'][index]
    #     predicted = prediction_liblinear[index]
    #     # print("ID : " + str(id) + " Predicted Tag : " + str(predicted))
    #     output.write(str(id) + "\t" + predicted)
    #     output.write("\n")
