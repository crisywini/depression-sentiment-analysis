import pandas as pd
from bs4 import BeautifulSoup
import re
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


def main():
    cols = ["id", "message", "label"]
    data = pd.read_csv("/home/crisisanchezp/uniquindio/Trabajo_de_grado/datasets/sentiment_tweets3.csv",
                       header=None,
                       names=cols,
                       engine='python',
                       encoding='latin1')
    data.drop(['id'],
              axis=1,
              inplace=True)
    data_clean = [clean_tweet(tweet) for tweet in data.message]

    print("Process finished")

if __name__ == '__main__':
    main()
