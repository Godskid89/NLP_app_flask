# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:28:15 2019

@author: Joseph.Oladokun
"""

from flair.data import Sentence
import pandas as pd
from flair.models import TextClassifier
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.trainers import ModelTrainer
from pathlib import Path

## airliner_nlp = TextClassifier.load('en-sentiment')


## Import Data
data = pd.read_csv("Tweets.csv", encoding = 'latin-1').sample(frac=1).drop_duplicates()


data['airline_sentiment'] = '__airline_sentiment__' + data['airline_sentiment'].astype(str)

data.iloc[0:int(len(data)*0.8)].to_csv('train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('dev.csv', sep='\t', index = False, header = False)


corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./'), test_file='train.csv', dev_file='dev.csv', train_file='test.csv')

word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]

document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)

trainer = ModelTrainer(classifier, corpus)

trainer.train('./', max_epochs=2)