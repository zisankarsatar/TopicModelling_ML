# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:27:55 2020

@author: ZİŞAN
"""
#kaynak:https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
#veri seti 15 yıl boyunca yayınlanan bir milyondan fazla haber başlığının listesidir
import pandas as pd
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

#verilere bakalım
print(len(documents))
print(documents[:5])

#önişleme
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

#Kök önişleme
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#Önişlemeden sonra önizlemek için bir belge seçin .
doc_sample = documents[documents['index'] == 4310].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#Başlık metnini önceden işleyerek sonuçları 'işlenmiş_docs' olarak kaydedin
processed_docs = documents['headline_text'].map(preprocess)
processed_docs[:10]

#'processed_docs' öğesinden bir eğitim kümesinde bir kelimenin kaç kez görüneceğini içeren bir sözlük oluşturun.
#bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
#Ekranda görünen belirteçleri filtrelem
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

#Her belge için bu kelimelerin kaç kelime ve kaç kez göründüğünü bildiren bir sözlük oluşturuyoruz 
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4310]

#Önceden İşlenmiş Belgemiz için Bag Of Words kelimesini önizleyin
bow_doc_4310 = bow_corpus[4310]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
          dictionary[bow_doc_4310[i][0]], bow_doc_4310[i][1]))
    
#TF-IDF
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

#Bag of Words kullanrak LDA
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)  

#Her konu için, o konudaki kelimeleri ve göreceli ağırlığını araştıracağız.
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    

#TF-IDF kullanarak LDA çalıştırma
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
    
#LDA Bag of Words modelini kullanarak örnek belgeyi sınıflandırarak performans değerlendirmesi
processed_docs [4310]

for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
    

#LDA TF-IDF modelini kullanarak örnek belgeyi sınıflandırarak performans değerlendirmesi.
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

#Görünmeyen belgede test modeli
unseen_document = 'How a Pentagon deal became an identity crisis for Google'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))