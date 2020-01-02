# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 23:05:42 2020

@author: ZİŞAN
"""

import pandas as pd
import numpy as np

#20000 tanesi ile işlem yapıyoruz
reviews_datasets = pd.read_csv(r'abcnews-date-text.csv')
reviews_datasets = reviews_datasets.head(20000)
reviews_datasets= pd.concat([reviews_datasets.headline_text],axis=1)
reviews_datasets.dropna()

#verileri görelim
reviews_datasets.head()

#köklerine ayırıyor
from sklearn.feature_extraction.text import CountVectorizer 

#Yalnızca belgenin% 80'inden daha azında görünen ve en az 2 belgede görünen kelimeleri dahil etmeyi belirtiyoruz.
count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(reviews_datasets.values.astype('U').ravel())

#Şimdi belge terim matrisimize bakalım:
doc_term_matrix

# metnimizin bölünmesini istediğimiz kategori veya konu sayısını belirtir
from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)

import random
#kelime dağarcığımızdan rastgele 10 kelime getirir
#get_feature_names()Yöntemi kullanabilir ve getirmek istediğimiz kelimenin kimliğini iletebiliriz.
for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])
   
#ilk konu için yüksek olalıklı 10 kelimeyi getirelim
first_topic = LDA.components_[0]

#en yüksek olasılıklı 10 kelime dizinin son 10 dizinine ait olacaktır.
#Endeksleri olasılık değerlerine göre sıralamak için argsort()fonksiyonu kullanabiliriz . 
#Sıralandığında, en yüksek olasılıklı 10 kelime dizinin son 10 dizinine ait olacaktır. Aşağıdaki 
#komut dosyası, en yüksek olasılıklı 10 sözcüğün dizinlerini döndürür:
top_topic_words = first_topic.argsort()[-10:]
print(top_topic_words)

#Bu dizinler daha sonra count_vectnesneden kelimelerin değerini almak için kullanılabilir ve bu şekilde yapılabilir:
for i in top_topic_words:
    print(count_vect.get_feature_names()[i])
    
#Beş konunun tümü için en yüksek olasılıklı 10 kelimeyi basalım:    
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')
    
    
#orijinal veri çerçevesine metnin konusunu saklayacak bir sütun ekleyeceğiz
topic_values = LDA.transform(doc_term_matrix)
#her bir sütunun belirli bir konunun olasılık değerine karşılık geldiği yerde 5 sütuna sahip olduğu anlamına gelir
topic_values.shape

#veri çerçevesindeki konu için yeni bir sütun ekler ve konu değerini sütundaki her satıra atar:
reviews_datasets['Topic'] = topic_values.argmax(axis=1)
reviews_datasets.head()

