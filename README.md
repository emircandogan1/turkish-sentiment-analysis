# turkish-sentiment-analysis
This project involves analyzing Turkish tweets and categorizing them into three different sentiment categories: positive, negative, and neutral. Machine learning models are utilized to classify the sentiment of the tweets.

![neutral](https://github.com/emircandogan1/turkish-sentiment-analysis/assets/69003695/d3c637c0-93c5-407a-b4a8-e430df89376e)


## files
* normalization.py is the Python script intended for preprocessing data, specifically aimed at normalization. Part of the NLP pipeline utilizes the 'Zemberek' (https://github.com/ahmetaa/zemberek-nlp) library and its adjunct 'Zeyrek' for certain stages, while the remaining text cleaning processes, such as removing punctuations, stopwords, badwords, etc., are handled manually. 
* analyzer.py contains text analysis functionalities post-normalization, identifying the words present in the text, generating pie charts, and highlighting the most frequently occurring words etc..
* model.py includes Logistic Regression, Support Vector Machines, Multinomial Naive Bayes, and Random Forest models. After training these models with the text data, it performs testing, ranks the models based on their F1 scores, and saves the models for deployment.

## usage
* normalization.py requires three things: a dataset (in xlsx format, with one sentence per row), stopwords, and checkwords (checkwords are used to remove sentences containing undesirable words). After running the file, the normalized text is saved to your disk in xlsx format.
* analyzer.py requires the dataset saved by normalization.py
* model.py requires dataset (normalized data), stopwords.

## License 
GNU
