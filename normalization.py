import pandas as pd
import re
import string
from nltk.corpus import stopwords
import logging
logging.getLogger("zemberek.morphology.turkish_morphology").setLevel(logging.ERROR)
logging.getLogger("zeyrek").setLevel(logging.ERROR)
from zemberek import (
    TurkishSentenceNormalizer,
    TurkishMorphology,
    TurkishTokenizer
)
import zeyrek
import json
import time

class TweetNormalizer:
    def __init__(self, file_path, stopwords_path, checkwords_path):
        self.df = pd.read_excel(file_path).drop(columns=['Unnamed: 0'])
        self.stop_words = open(stopwords_path, 'r', encoding='utf-8').read().split()
        self.stop_words += stopwords.words("turkish")
        with open(checkwords_path, 'r', encoding='utf-8') as f:
            check_words_data = json.load(f)
            self.check_words = check_words_data.get("check_words", [])
        self.morphology = TurkishMorphology.create_with_defaults()
        self.normalizer = TurkishSentenceNormalizer(self.morphology)

    def normalize_text(self):
        copy_df = self.df.copy()
        for index, sentence in copy_df['full_text'].items():
            normalized_text = self.normalizer.normalize(sentence)
            copy_df.loc[index, 'full_text'] = normalized_text
        self.df = copy_df

    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emotions
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoji_pattern, '', text)

    def clean_text(self):
        self.df['full_text'] = self.df["full_text"].apply(self.remove_emojis)   # remove emojis
        self.df['full_text'] = self.df['full_text'].apply(self.remove_mentions) # remove mentions
        self.df = self.df[self.df['full_text'].str.split().apply(len) > 2]  # remove single words
        self.df = self.df.drop_duplicates(subset=['full_text']) # drop duplicates
        self.df['full_text'] = self.df['full_text'].apply(lambda x: " ".join(word for word in x.split() if word not in self.stop_words)) # remove stopwords
        mask = self.df['full_text'].str.contains('|'.join(self.check_words))    # remove check_words if contains # try without case, na
        self.df = self.df[~mask]
        self.df.reset_index(drop=True, inplace=True)

    def remove_mentions(self, text):
        text = re.sub(r'@\w+\s*', '', text)                      
        text = re.sub(r'#\w+\s*', '', text)                      
        text = re.sub(r'https?://\S+', '', text)                
        text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit() and char not in ['‘', '’', '“', '”', '…']])  
        text = text.lower()
        text = ' '.join(text.split())   
        return text

    def tokenize_text(self, text):
        tokenizer = TurkishTokenizer.DEFAULT
        tokens = tokenizer.tokenize(text)
        return [token.content for token in tokens]  

    def lemmatize_text(self):
        sentences = self.df['tokenized_text'].copy()

        for token in sentences:
            j = 0
            for word in token:
                new_word = word.replace('"', '').replace("’", '').replace("'", '').replace("”", '')
                token[j] = new_word
                j += 1

        analyzer = zeyrek.MorphAnalyzer()
        lem_sent = []
        for sent in sentences:
            normalized_sent = []
            for word in sent:
                if word == '':
                    continue
                else:
                    lem_word = analyzer.lemmatize(word)
                    normalized_sent.append(lem_word[0][1][0])
            lem_sent.append(normalized_sent)

        lem_sent = [[token.lower() for token in sent] for sent in lem_sent]
        lem_sent = [sent for sent in lem_sent if sent]  
        self.df['lem_sent'] = lem_sent
        self.df['lem_sent'] = self.df['lem_sent'].apply(' '.join)

    def process_text(self):
        self.df['tokenized_text'] = self.df['full_text'].apply(self.tokenize_text)
        self.lemmatize_text()

    def save_dataframe(self, output_path):
        self.df = self.df.drop(columns=['full_text', 'tokenized_text'], axis=1)
        self.df = self.df.rename(columns={'lem_sent': 'tweet', 'label': 'sentiment'})
        self.df.to_excel(output_path, index=False)

    def run(self, output_path):
        print("Data Preprocessing işlemi başladı, lütfen bekleyin.")
        start_time = time.time()

        self.normalize_text()
        self.clean_text()
        self.process_text()
        self.save_dataframe(output_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60
        print("İşlemler tamamlandı, veri seti normalleştirildi. 'normalized_data' olarak normalleştirilmiş metin diskinize kaydedildi.")
        print(f"Geçen Süre: {elapsed_time:.2f} saniye. ({elapsed_minutes:.2f} dakika)")

if __name__ == "__main__":
    file_path = input("Normalleştirilmek istenen metin veri yolu: ")    # Data path for normalize
    stopwords_path = input("Stopwords dosyası veri yolu: ")     # Stopwords path
    checkwords_path = input("Check words dosyasının yolunu girin: ")    # Unnecesarray words in your texts
    
    normalizer = TweetNormalizer(
        file_path=file_path, 
        stopwords_path=stopwords_path, 
        checkwords_path=checkwords_path
    )
    normalizer.run('normalized_data.xlsx')
