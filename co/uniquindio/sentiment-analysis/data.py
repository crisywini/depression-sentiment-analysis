from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from feature_extraction import get_bow_sklearn, get_tf_idf_sklearn


def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, 'lxml').get_text()  # Decode because is in lxml format by Twitter
    # Remove users
    tweet = re.sub(r'@[A-Za-z0-9]+', ' ', tweet)
    # Remove URLs
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', ' ', tweet)
    # Keep A-Za-z
    tweet = re.sub(r"@[^a-zA-Z.!?']+", ' ', tweet)
    # Remove blank spaces
    tweet = re.sub(r' +', ' ', tweet)
    return tweet


def remove_stop_words(sentence, stop_words):
    return [word for word in sentence if word.upper() not in stop_words]


def tokenize_without_grammatical_characters(sentence, tokenizer):
    word_tokens = tokenizer.tokenize(str(sentence))
    return [word.lower() for word in word_tokens]


def remove_non_ascii(text):
    return "".join([word for word in text if ord(word) < 128])


def gensim_preprocess_data(data, punctuation):
    sentences = sent_tokenize(data)
    tokenized_sentences = list([word_tokenize(sentence) for
                                sentence in sentences])
    stop_words = [word.upper() for word in stopwords.words('english')]
    for i in range(0, len(tokenized_sentences)):
        tokenized_sentences[i] = [word for word in tokenized_sentences[i] if
                                  (word not in punctuation and word.upper() not in stop_words)]
    print("Tokenized words are")
    print(tokenized_sentences)
    return tokenized_sentences


def gensim_skip_gram(data):
    # sentences = gensim_preprocess_data(data, punctuation)
    skip_gram = Word2Vec(sentences=data, window=1,
                         min_count=2, sg=1)
    # word_embedding = list(skip_gram.wv.vocab)
    # word_embedding = skip_gram[skip_gram.wv.key_to_index]
    vocab = skip_gram.wv.index_to_key
    # skip_gram.wv.index_to_key = [remove_non_ascii(text) for text in skip_gram.wv.index_to_key]
    # skip_gram.wv.index_to_key = [text for text in skip_gram.wv.index_to_key if len(text)>0]
    print("Vocab")
    print(vocab)
    vector = skip_gram.wv['depression']
    print("Depresison")
    print(vector)
    print("Most similar to depression")
    print(skip_gram.wv.most_similar("depression"))
    word_embedding = skip_gram.wv[skip_gram.wv.index_to_key]
    print("All the emeddings")
    print(word_embedding)

    pca = PCA(n_components=2)
    word_embedding = pca.fit_transform(word_embedding)
    # Plotting results from trained word embedding
    plt.scatter(word_embedding[:, 0], word_embedding[:, 1])
    word_list = list(skip_gram.wv.index_to_key)
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(word_embedding[i, 0],
                               word_embedding[i, 1]))
    plt.show()


def main():
    cols = ["id", "message", "label"]
    data = pd.read_csv("sentiment_tweets3.csv",
                       header=None,
                       names=cols,
                       engine='python',
                       encoding='latin1')
    data.drop(['id'],
              axis=1,
              inplace=True)
    data_clean = [clean_tweet(tweet) for tweet in data.message.values]

    tokens = [word_tokenize(tweet) for tweet in data_clean]  # NxRows

    stop_words = [word.upper() for word in stopwords.words('english')]
    word_tokens = [remove_stop_words(sentence, stop_words) for sentence in tokens]
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = [tokenize_without_grammatical_characters(sentence, tokenizer) for sentence in word_tokens]
    # word_tokens = [remove_non_ascii(word) for word in word_tokens]
    # x, y = get_bow_sklearn(word_tokens[0])

    # print(x, y)
    # x = get_tf_idf_sklearn(word_tokens[0])
    # print("TF/IDF")
    # print(x)
    print("Process finished")

    gensim_skip_gram(word_tokens)


if __name__ == '__main__':
    main()
