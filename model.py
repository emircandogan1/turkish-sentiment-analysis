import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import time
import pickle

# Ensure you have the nltk stopwords
# nltk.download('stopwords')

class SentimentModels:
    def __init__(self, data_path, stopwords_path):
        self.data_path = data_path
        self.stopwords_path = stopwords_path
        self.final_stopwords = self.load_stopwords()
        self.data = self.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.important_words = {'iyi': 3.0, 'kötü': 3.0, 'ancak': 2.0, 'fakat': 2.0}
        self.f1_scores = {}  

    def load_data(self):
        return pd.read_excel(self.data_path)

    def load_stopwords(self):
        stop_words = open(self.stopwords_path, 'r', encoding='utf-8').read().split()
        sw = stopwords.words("turkish")
        return sw + stop_words

    def split_data(self):
        return train_test_split(self.data.tweet, self.data.sentiment, test_size=0.20, random_state=30)

    def apply_custom_weights(self, X_tfidf, feature_names):
        for word, weight in self.important_words.items():
            if word in feature_names:
                idx = feature_names.tolist().index(word)
                X_tfidf[:, idx] = X_tfidf[:, idx].multiply(weight)
        return X_tfidf

    def train_model(self, clf, param_grid, model_name):
        vectorizer = TfidfVectorizer(stop_words=self.final_stopwords, ngram_range=(1, 2))
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)
        ])
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        best_vectorizer = grid_search.best_estimator_.named_steps['vectorizer']
        best_clf = grid_search.best_estimator_.named_steps['clf']
        feature_names = best_vectorizer.get_feature_names_out()

        X_train_tfidf = best_vectorizer.transform(self.X_train)
        X_train_tfidf = self.apply_custom_weights(X_train_tfidf, feature_names)
        best_clf.fit(X_train_tfidf, self.y_train)

        best_score = grid_search.best_score_
        self.f1_scores[model_name] = best_score

        return best_vectorizer, best_clf

    def evaluate_model(self, vectorizer, clf):
        X_test_tfidf = vectorizer.transform(self.X_test)
        X_test_tfidf = self.apply_custom_weights(X_test_tfidf, vectorizer.get_feature_names_out())
        y_pred = clf.predict(X_test_tfidf)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

    def train_logistic_regression(self):
        param_grid = {
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1, 10]
        }
        return self.train_model(LogisticRegression(), param_grid, "Logistic Regression")

    def train_multinomial_nb(self):
        param_grid = {
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'clf__alpha': [0.1, 1, 10]
        }
        return self.train_model(MultinomialNB(), param_grid, "MultinomialNB")

    def train_random_forest(self):
        param_grid = {
            'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'vectorizer__max_df': [0.75, 0.85, 0.95],
            'vectorizer__min_df': [1, 2, 3],
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 10, 20, 30]
        }
        return self.train_model(RandomForestClassifier(random_state=30), param_grid, "Random Forest")

    def train_svm(self):
        param_grid = {
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1, 10]
        }
        return self.train_model(SVC(kernel='linear'), param_grid, "SVM")

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"\nModel '{filename}' isminde diskinize kaydedildi.")

if __name__ == "__main__":
    print("\nModeller train ve test edilmeye başlandı, lütfen bekleyiniz.")
    start_time = time.time()
    data_path = r'normalized_data.xlsx'
    stopwords_path = 'stopwords.txt'
    model = SentimentModels(data_path, stopwords_path)

    print("\n### Logistic Regression ###")
    vectorizer, clf = model.train_logistic_regression()
    model.evaluate_model(vectorizer, clf)
    save_model((vectorizer, clf), "logistic_regression_model.pkl")

    print("\n### MultinomialNB ###")
    vectorizer, clf = model.train_multinomial_nb()
    model.evaluate_model(vectorizer, clf)
    save_model((vectorizer, clf), "multinomial_nb_model.pkl")

    print("\n### Random Forest ###")
    vectorizer, clf = model.train_random_forest()
    model.evaluate_model(vectorizer, clf)
    save_model((vectorizer, clf), "random_forest_model.pkl")

    print("\n### SVM ###")
    vectorizer, clf = model.train_svm()
    model.evaluate_model(vectorizer, clf)
    save_model((vectorizer, clf), "svm_model.pkl")
    
    end_time = time.time()
    elapsed_time = end_time-start_time
    elapsed_minute = elapsed_time / 60

    print("\nModellerin eğitimi tamamlandı, sırasıyla en iyi modeller:\n ")
    
    for model_name, score in sorted(model.f1_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"- {model_name}, F1 Skoru = {score * 100:.2f}%")

    print(f"\nGeçen Süre: {elapsed_time:.2f} saniye. ({elapsed_minute:.2f} dakika)")

