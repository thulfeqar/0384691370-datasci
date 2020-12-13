import data_processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

def binaryNB(df):
    #declares
    x_train, x_test, y_train, y_test = train_test_split(df['filteredText'],df['label'])
    bin_vec = CountVectorizer(binary=True)
    model_bin = BernoulliNB()

    #shape data
    bin_vec.fit(x_train)
    x_train_bin = bin_vec.transform(x_train)
    x_test_bin = bin_vec.transform(x_test)

    #run model
    model_bin.fit(x_train_bin,y_train)
    preds = model_bin.predict(x_test_bin)
    accuracy = (preds==y_test).mean()
    print(accuracy)
    return 1

def main():
    df = data_processing.loadFile('./train_limited.csv')  # Enter file Path
    df_clean = data_processing.cleanFile(df)
    #['id', 'title', 'author', 'text', 'label', 'filteredText']
    binaryNB(df_clean)


if __name__ == "__main__":
    main()