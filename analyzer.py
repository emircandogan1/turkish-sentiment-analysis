import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import time

class TweetAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_excel(data_path)
        self.old_characters = 'çğıöşüÇĞİÖŞÜ'
        self.new_characters = 'cgiosuCGIOSU'
        self.translate_tweets()
    
    def translate_tweets(self):
        translation_table = str.maketrans(self.old_characters, self.new_characters)
        self.df['tweet'] = self.df['tweet'].apply(lambda x: x.translate(translation_table))

    def calculate_word_freq(self):
        all_words = [word for tweet in self.df['tweet'] for word in tweet.split()]
        self.word_freq = Counter(all_words)

    def print_word_count_info(self):
        self.df['word_count'] = self.df['tweet'].apply(lambda x: len(x.split()))
        total_word_count = self.df['word_count'].sum()
        average_word_count = total_word_count / len(self.df)
        print(f"\nTotal Word Count: {total_word_count}")
        print(f"Average Word Count Per Tweet: {average_word_count:.2f}")

    def print_balance_info(self):
        tweet_list = self.df['tweet']
        negative_list = self.df[self.df['sentiment'] == 0]
        positive_list = self.df[self.df['sentiment'] == 1]
        neutral_list = self.df[self.df['sentiment'] == 2]

        print(f"\n### Tweets Count ###\n\nTotal Tweets: {len(tweet_list)}")
        print(f"Positive Tweets: {len(positive_list)}")
        print(f"Negative Tweets: {len(negative_list)}")
        print(f"Neutral Tweets: {len(neutral_list)}\n")
        print(f"### Percentile ###\n")
        print(f"Positive Tweets Percent: {len(positive_list) / len(tweet_list) * 100:.2f} %")
        print(f"Negative Tweets Percent: {len(negative_list) / len(tweet_list) * 100:.2f} %")
        print(f"Neutral Tweets Percent: {len(neutral_list) / len(tweet_list) * 100:.2f} %")

    def plot_balance_info(self):
        tweet_list = self.df['tweet']
        positive_list = self.df[self.df['sentiment'] == 1]
        negative_list = self.df[self.df['sentiment'] == 0]
        neutral_list = self.df[self.df['sentiment'] == 2]

        positive = int(100 * len(positive_list) / len(tweet_list))
        negative = int(100 * len(negative_list) / len(tweet_list))
        neutral = int(100 * len(neutral_list) / len(tweet_list))

        labels = [f'Positive [{positive}%]', f'Negative [{negative}%]', f'Neutral [{neutral}%]']
        sizes = [positive, negative, neutral]
        colors = ['green', 'red', 'blue']

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, colors=colors, startangle=90, autopct='%1.1f%%', pctdistance=0.85)
        plt.legend(labels, loc="best")
        plt.title("Balance Between Sentiments")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def get_top_n_words(self, corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_word_frequency(self, top_n=20):
        common_words = self.get_top_n_words(self.df['tweet'], top_n)
        common_df = pd.DataFrame(common_words, columns=['word', 'count'])

        plt.figure(figsize=(10, 6))
        common_df.groupby('word').sum()['count'].sort_values(ascending=False).plot(
            kind='bar', color='blue', edgecolor='black')
        plt.xlabel("Top Words")
        plt.ylabel("Count")
        plt.title("Word Frequency - Most Popular Words")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Text Analyzer başladı.")
    start_time = time.time()

    analyzer = TweetAnalyzer('normalized_data.xlsx')
    analyzer.calculate_word_freq()
    analyzer.print_word_count_info()
    analyzer.print_balance_info()
    analyzer.plot_balance_info()
    analyzer.plot_word_frequency(top_n=20)
    
    end_time = time.time()
    elapsed_time = end_time-start_time
    elapsed_minutes = elapsed_time / 60
    
    print("\nText Analyzer tamamlandı.")
    print(f"Geçen Süre: {elapsed_time:.2f} saniye. ({elapsed_minutes:.2f} dakika)")
