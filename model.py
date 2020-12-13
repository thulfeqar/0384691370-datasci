import data_processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

#--------------WORD2VEC---------------

def word2vecNB(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    #format data for gensim
    sents = []
    for indx, row in x_train.items():
        sents.append(TaggedDocument(words=row.split(), tags=str(indx)))

    model = Doc2Vec(sents)

    #vectorize x data
    x_train_vec = x_train.apply(lambda row: model.infer_vector(row.split()))
    x_train_df = x_train_vec.apply(pd.Series)

    x_test_vec = x_test.apply(lambda row: model.infer_vector(row.split()))
    x_test_df = x_test_vec.apply(pd.Series)

    #test in model
    model_bin = BernoulliNB()
    model_bin.fit(x_train_df, y_train)
    preds = model_bin.predict(x_test_df)

    # scoring
    accuracy = (preds == y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("W2V naive bayes")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    return 1

#--------------SVM MODELS-------------

def binarySVM(X,Y):
    # declares
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    bin_vec = CountVectorizer(binary=True)
    model = svm.SVC()

    # shape data
    bin_vec.fit(x_train)
    x_train_bin = bin_vec.transform(x_train)
    x_test_bin = bin_vec.transform(x_test)

    # run model
    model.fit(x_train_bin, y_train)
    preds = model.predict(x_test_bin)

    # scoring
    accuracy = (preds == y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("binary SVM")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    return 1

def tfidfSVM(X,Y):
    #declares
    x_train, x_test, y_train, y_test = train_test_split(X,Y)
    tf_vec = TfidfVectorizer(ngram_range=(1, 2))
    model = svm.SVC()

    #shape data
    tf_vec.fit(x_train)
    x_train_tf = tf_vec.transform(x_train)
    x_test_tf = tf_vec.transform(x_test)

    #run model
    model.fit(x_train_tf,y_train)
    preds = model.predict(x_test_tf)

    #scoring
    accuracy = (preds==y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("TF in DF SVM")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    return 1

#---------------------NAIVE BAYES MODELS------------------

def binaryNB(X,Y):
    #declares
    x_train, x_test, y_train, y_test = train_test_split(X,Y)
    bin_vec = CountVectorizer(binary=True)
    model_bin = BernoulliNB()

    #shape data
    bin_vec.fit(x_train)
    x_train_bin = bin_vec.transform(x_train)
    x_test_bin = bin_vec.transform(x_test)

    #run model
    model_bin.fit(x_train_bin,y_train)
    preds = model_bin.predict(x_test_bin)

    #scoring
    accuracy = (preds==y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("binary naive bayes")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    return 1

def tfidfNB(X,Y):
    #declares
    x_train, x_test, y_train, y_test = train_test_split(X,Y)
    tf_vec = TfidfVectorizer(ngram_range=(1, 2))
    model = BernoulliNB()

    #shape data
    tf_vec.fit(x_train)
    x_train_tf = tf_vec.transform(x_train)
    x_test_tf = tf_vec.transform(x_test)

    #run model
    model.fit(x_train_tf,y_train)
    preds = model.predict(x_test_tf)

    #scoring
    accuracy = (preds==y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("TF in DF naive bayes")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    return 1

#----------------------MAIN----------------------


def main():
    df = data_processing.loadFile('./train_limited.csv')  # Enter file Path
    df_clean = data_processing.cleanFile(df)
    #['id', 'title', 'author', 'text', 'label', 'filteredText']

    #baseline models here
    print("----baseline data----")
    binaryNB(df_clean['filteredText'],df_clean['label'])
    tfidfNB(df_clean['filteredText'], df_clean['label'])
    binarySVM(df_clean['filteredText'],df_clean['label'])
    tfidfSVM(df_clean['filteredText'], df_clean['label'])
    word2vecNB(df_clean['filteredText'], df_clean['label'])

    #models with undersampler
    print("----undersampled data----")
    undersampler = RandomUnderSampler()
    x_rus, y_rus = undersampler.fit_sample(df_clean.drop(columns=['label']),df_clean['label'])
    binaryNB(x_rus['filteredText'], y_rus)
    tfidfNB(x_rus['filteredText'], y_rus)
    binarySVM(x_rus['filteredText'], y_rus)
    tfidfSVM(x_rus['filteredText'], y_rus)

    #models with oversampler
    print("----oversampled data----")
    oversampler = RandomOverSampler()
    x_ros, y_ros = oversampler.fit_sample(df_clean.drop(columns=['label']), df_clean['label'])
    binaryNB(x_ros['filteredText'], y_ros)
    tfidfNB(x_ros['filteredText'], y_ros)
    binarySVM(x_ros['filteredText'], y_ros)
    tfidfSVM(x_ros['filteredText'], y_ros)



if __name__ == "__main__":
    main()