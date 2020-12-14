import data_processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import SentimentAnalysisPY
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve

#-----------------KMEANS---------------

def KMeanLayer(x_train, x_test, y_train, y_test):
    modelK = KMeans(n_clusters=2)
    modelK.fit(x_train)
    cluster = modelK.predict(x_test)
    accuracy = (cluster==y_test).mean()
    roc_auc = roc_auc_score(y_test, cluster)
    print("k means layer")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))
    return cluster

#--------------WORD2VEC---------------

def word2vecNB(x_train, x_test, y_train, y_test):

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
    probs = model_bin.predict_proba(x_test_df)

    # scoring
    accuracy = (preds == y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("W2V naive bayes")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    label = type(model_bin).__name__
    prec, rec, thresholds = precision_recall_curve(y_test, probs[:,1])

    return preds, fpr, tpr, label, prec, rec

def word2vecSVM(x_train, x_test, y_train, y_test):

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
    model = svm.SVC(probability=True)
    model.fit(x_train_df, y_train)
    preds = model.predict(x_test_df)
    probs = model.predict_proba(x_test_df)

    # scoring
    accuracy = (preds == y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("W2V SVM")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    label = type(model).__name__

    prec, rec, thresholds = precision_recall_curve(y_test, probs[:,1])

    return preds, fpr, tpr, label, prec, rec
#--------------SVM MODELS-------------

def binarySVM(x_train, x_test, y_train, y_test):
    # declares
    bin_vec = CountVectorizer(binary=True)
    model = svm.SVC(probability=True)

    # shape data
    bin_vec.fit(x_train)
    x_train_bin = bin_vec.transform(x_train)
    x_test_bin = bin_vec.transform(x_test)

    # run model
    model.fit(x_train_bin, y_train)
    preds = model.predict(x_test_bin)
    probs = model.predict_proba(x_test_bin)
    # scoring
    accuracy = (preds == y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("binary SVM")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    label = type(model).__name__

    prec, rec, thresholds = precision_recall_curve(y_test, probs[:,1])

    return preds, fpr, tpr, label, prec, rec

def tfidfSVM(x_train, x_test, y_train, y_test):
    #declares
    tf_vec = TfidfVectorizer(ngram_range=(1, 2))
    model = svm.SVC(probability=True)

    #shape data
    tf_vec.fit(x_train)
    x_train_tf = tf_vec.transform(x_train)
    x_test_tf = tf_vec.transform(x_test)

    #run model
    model.fit(x_train_tf,y_train)
    preds = model.predict(x_test_tf)
    probs = model.predict_proba(x_test_tf)

    #scoring
    accuracy = (preds==y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("TF in DF SVM")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    label = type(model).__name__

    prec, rec, thresholds = precision_recall_curve(y_test, probs[:,1])

    return preds, fpr, tpr, label, prec, rec


#---------------------NAIVE BAYES MODELS------------------

def binaryNB(x_train, x_test, y_train, y_test):
    #declares
    bin_vec = CountVectorizer(binary=True)
    model_bin = BernoulliNB()

    #shape data
    bin_vec.fit(x_train)
    x_train_bin = bin_vec.transform(x_train)
    x_test_bin = bin_vec.transform(x_test)

    #run model
    model_bin.fit(x_train_bin,y_train)
    preds = model_bin.predict(x_test_bin)
    probs = model_bin.predict_proba(x_test_bin)


    #scoring
    accuracy = (preds==y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("binary naive bayes")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    label = type(model_bin).__name__

    prec, rec, thresholds = precision_recall_curve(y_test, probs[:,1])

    return preds, fpr, tpr, label, prec, rec


def tfidfNB(x_train, x_test, y_train, y_test):
    #declares
    tf_vec = TfidfVectorizer(ngram_range=(1, 2))
    model = BernoulliNB()

    #shape data
    tf_vec.fit(x_train)
    x_train_tf = tf_vec.transform(x_train)
    x_test_tf = tf_vec.transform(x_test)

    #run model
    model.fit(x_train_tf,y_train)
    preds = model.predict(x_test_tf)
    probs = model.predict_proba(x_test_tf)

    #scoring
    accuracy = (preds==y_test).mean()
    roc_auc = roc_auc_score(y_test, preds)
    print("TF in DF naive bayes")
    print("accuracy: " + str(accuracy))
    print("ROC-AUC: " + str(roc_auc))

    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    label = type(model).__name__

    prec, rec, thresholds = precision_recall_curve(y_test, probs[:,1])

    return preds, fpr, tpr, label, prec, rec


#----------------------MAIN----------------------


def main():
    df_clean = data_processing.loadFile('./xaa.csv')  # Enter file Path
    print(df_clean.shape)
    df_clean = data_processing.cleanFile(df_clean)
    print("cleaned")
    df_clean = data_processing.authorScore(df_clean)
    print("scored")
    df_clean = SentimentAnalysisPY.calculateSentiment(df_clean)

    #['id', 'title', 'author', 'text', 'label', 'filteredText']

    #baseline models here
    print("----baseline data----")
    x_train_base, x_test_base, y_train_base, y_test_base = train_test_split(df_clean.drop(['label'],axis=1),df_clean['label'])
    preds1, fpr1, tpr1, label1, prec1, rec1 = binaryNB(x_train_base['filteredText'], x_test_base['filteredText'], y_train_base, y_test_base)
    preds1, fpr2, tpr2, label2, prec2, rec2 = tfidfNB(x_train_base['filteredText'], x_test_base['filteredText'], y_train_base, y_test_base)
    preds1, fpr3, tpr3, label3, prec3, rec3 = binarySVM(x_train_base['filteredText'], x_test_base['filteredText'], y_train_base, y_test_base)
    preds1, fpr4, tpr4, label4, prec4, rec4 = tfidfSVM(x_train_base['filteredText'], x_test_base['filteredText'], y_train_base, y_test_base)
    preds1, fpr5, tpr5, label5, prec5, rec5 = word2vecNB(x_train_base['filteredText'], x_test_base['filteredText'], y_train_base, y_test_base)
    preds1, fpr6, tpr6, label6, prec6, rec6 = word2vecSVM(x_train_base['filteredText'], x_test_base['filteredText'], y_train_base, y_test_base)

    plt.plot(fpr1, tpr1, label="Binary Naive Bayes")
    plt.plot(fpr2, tpr2, label="TFiDF Naive Bayes")
    plt.plot(fpr3, tpr3, label="Binary SVM")
    plt.plot(fpr4, tpr4, label="TFiDF SVM")
    plt.plot(fpr5, tpr5, label="W2V Naive Bayes")
    plt.plot(fpr6, tpr6, label="W2V SVM")
    plt.legend(loc='best')
    plt.savefig('Baseline - ROC.png')
    plt.clf()

    plt.plot(rec1, prec1, label="Binary Naive Bayes")
    plt.plot(rec2, prec2, label="TFiDF Naive Bayes")
    plt.plot(rec3, prec3, label="Binary SVM")
    plt.plot(rec4, prec4, label="TFiDF SVM")
    plt.plot(rec5, prec5, label="W2V Naive Bayes")
    plt.plot(rec6, prec6, label="W2V SVM")
    plt.legend(loc='best')
    plt.savefig('Baseline - Precision Recall Curve.png')
    plt.clf()

    #models with undersampler
    print("----undersampled data----")
    undersampler = RandomUnderSampler()
    x_rus, y_rus = undersampler.fit_sample(df_clean.drop(columns=['label']),df_clean['label'])
    x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus,y_rus)
    preds1, fpr1, tpr1, label1, prec1, rec1 = word2vecSVM(x_train_rus['filteredText'], x_test_rus['filteredText'], y_train_rus, y_test_rus)
    preds1, fpr2, tpr2, label2, prec2, rec2 = binaryNB(x_train_rus['filteredText'], x_test_rus['filteredText'], y_train_rus, y_test_rus)
    preds1, fpr3, tpr3, label3, prec3, rec3 = tfidfNB(x_train_rus['filteredText'], x_test_rus['filteredText'], y_train_rus, y_test_rus)
    preds1, fpr4, tpr4, label4, prec4, rec4 = binarySVM(x_train_rus['filteredText'], x_test_rus['filteredText'], y_train_rus, y_test_rus)
    preds1, fpr5, tpr5, label5, prec5, rec5 = tfidfSVM(x_train_rus['filteredText'], x_test_rus['filteredText'], y_train_rus, y_test_rus)
    preds1, fpr6, tpr6, label6, prec6, rec6 = word2vecNB(x_train_rus['filteredText'], x_test_rus['filteredText'], y_train_rus, y_test_rus)

    plt.plot(fpr2, tpr2, label="Binary Naive Bayes")
    plt.plot(fpr3, tpr3, label="TFiDF Naive Bayes")
    plt.plot(fpr4, tpr4, label="Binary SVM")
    plt.plot(fpr5, tpr5, label="TFiDF SVM")
    plt.plot(fpr6, tpr6, label="W2V Naive Bayes")
    plt.plot(fpr1, tpr1, label="W2V SVM")
    plt.legend(loc='best')
    plt.savefig('Undersampled - ROC.png')
    plt.clf()

    plt.plot(rec2, prec2, label="Binary Naive Bayes")
    plt.plot(rec3, prec3, label="TFiDF Naive Bayes")
    plt.plot(rec4, prec4, label="Binary SVM")
    plt.plot(rec5, prec5, label="TFiDF SVM")
    plt.plot(rec6, prec6, label="W2V Naive Bayes")
    plt.plot(rec1, prec1, label="W2V SVM")
    plt.legend(loc='best')
    plt.savefig('Undersampled - Precision Recall Curve.png')
    plt.clf()

    #models with oversampler
    print("----oversampled data----")
    oversampler = RandomOverSampler()
    x_ros, y_ros = oversampler.fit_sample(df_clean.drop(columns=['label']), df_clean['label'])
    x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros,y_ros)

    preds1, fpr1, tpr1, label1, prec1, rec1 = binaryNB(x_train_ros['filteredText'], x_test_ros['filteredText'], y_train_ros, y_test_ros)
    preds1, fpr2, tpr2, label2, prec2, rec2 = tfidfNB(x_train_ros['filteredText'], x_test_ros['filteredText'], y_train_ros, y_test_ros)
    preds1, fpr3, tpr3, label3, prec3, rec3 = binarySVM(x_train_ros['filteredText'], x_test_ros['filteredText'], y_train_ros, y_test_ros)
    preds1, fpr4, tpr4, label4, prec4, rec4 = tfidfSVM(x_train_ros['filteredText'], x_test_ros['filteredText'], y_train_ros, y_test_ros)
    preds1, fpr5, tpr5, label5, prec5, rec5 = word2vecSVM(x_train_ros['filteredText'], x_test_ros['filteredText'], y_train_ros, y_test_ros)
    preds1, fpr6, tpr6, label6, prec6, rec6 = word2vecNB(x_train_ros['filteredText'], x_test_ros['filteredText'], y_train_ros, y_test_ros)

    plt.plot(fpr1, tpr1, label="Binary Naive Bayes")
    plt.plot(fpr2, tpr2, label="TFiDF Naive Bayes")
    plt.plot(fpr3, tpr3, label="Binary SVM")
    plt.plot(fpr4, tpr4, label="TFiDF SVM")
    plt.plot(fpr6, tpr6, label="W2V Naive Bayes")
    plt.plot(fpr5, tpr5, label="W2V SVM")
    plt.legend(loc='best')
    plt.savefig('Oversampled - ROC.png')
    plt.clf()

    plt.plot(rec1, prec1, label="Binary Naive Bayes")
    plt.plot(rec2, prec2, label="TFiDF Naive Bayes")
    plt.plot(rec3, prec3, label="Binary SVM")
    plt.plot(rec4, prec4, label="TFiDF SVM")
    plt.plot(rec6, prec6, label="W2V Naive Bayes")
    plt.plot(rec5, prec5, label="W2V SVM")
    plt.legend(loc='best')
    plt.savefig('Oversampled - Precision Recall Curve.png')
    plt.clf()

    #running through KMeans
    #first generate predictions for the entire corpus
    #then resplit x_ros
    #feed the predictions into KMeans

    preds , fpr, tpr, label, prec, rec = binaryNB(x_train_ros['filteredText'],x_ros['filteredText'],y_train_ros, y_ros)
    x_ros['k_in_preds'] = preds
    x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros)
    KMeanLayer(x_train_ros[['k_in_preds', 'authorScore', 'sentimentTitle', 'sentimentText']], x_test_ros[['k_in_preds', 'authorScore', 'sentimentTitle', 'sentimentText']], y_train_ros, y_test_ros)

    preds, fpr, tpr, label, prec, rec = binarySVM(x_train_ros['filteredText'], x_ros['filteredText'], y_train_ros, y_ros)
    x_ros['k_in_preds'] = preds
    x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros)
    KMeanLayer(x_train_ros[['k_in_preds', 'authorScore', 'sentimentTitle', 'sentimentText']], x_test_ros[['k_in_preds', 'authorScore', 'sentimentTitle', 'sentimentText']], y_train_ros, y_test_ros)


if __name__ == "__main__":
    main()