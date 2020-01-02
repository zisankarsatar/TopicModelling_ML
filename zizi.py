# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 02:11:11 2020

@author: ZİŞAN
"""

y=reviews_datasets['Topic'].values.reshape(-1,1)
x=reviews_datasets['Text']

import re #regular expression -cleaning data
import nltk#stopwords import edilir 
from nltk.corpus import stopwords
import nltk as nlp#kçklere ayırmak için lemmatizer import edilir
from sklearn.feature_extraction.text import CountVectorizer #bag of words yazdırmak için kullanılan vektör

nltk.download("stopwords")
lemma=nlp.WordNetLemmatizer()

X=[]
for d in x:
    d=re.sub("[^a-zA-Z]"," ",d)
    d= d.lower()
#    d = nltk.word_tokenize(d)
#    d= [word for word in d if not word in set(stopwords.words("english"))]
#    d=[lemma.lemmatize(word) for word in d]
#    d=" ".join(d)
    X.append(d)

max_features=5000;    
count_vectorizer=CountVectorizer(max_features=max_features) #parametre olarak stop_words="english" vb seyler alabilr
sparce_matrix=count_vectorizer.fit_transform(X).toarray()#x

son=sparce_matrix

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(son,y, test_size=0.2, random_state = 42)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

nb.fit(x_train,y_train)

y_pred=nb.predict(x_test)
print("accuracy: ",nb.score(y_test, y_pred))


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='utf-8').fit_transform()
#X_train_vectorized = vect.transform(x_train.ravel())

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
#başarım ölçülür
from sklearn.metrics import roc_auc_score
predictions = model.predict(count_vectorizer.fit_transform(x_test))
print('AUC: ', model.score(y_test, predictions))