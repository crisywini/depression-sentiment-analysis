from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import textblob
import numpy as np



def get_bow_sklearn(text):
    c = CountVectorizer(stop_words='english',
                        token_pattern=r'\w+')
    converted_data = c.fit_transform(text).todense()
    print(converted_data.shape)
    return converted_data, c.get_feature_names()


def term_frequency(word, textblob):
    return textblob.words.count(word) / float(len(textblob.words))


def document_counter(word, text):
    return sum(1 for blob in text if word in blob)


def idf(word, text):
    return np.log(len(text) / 1 + float(document_counter(word,
                                                         text)))


def tf_idf(word, blob, text):
    return term_frequency(word, blob) * idf(word, text)


def get_tf_idf_sklearn(document):
    t = TfidfVectorizer(stop_words='english',
                        token_pattern=r'\w+')
    x = t.fit_transform(document).todense()
    return x


def main():
    print("")


if __name__ == '__main__':
    main()
