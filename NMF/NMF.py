# Konu Modellemesi için NMF 
#Non-Negative Matrix Factorization (NMF) (Negatif Olmayan Matris Çarpanlarına Ayırma)

#Negatif olmayan matris çarpanlara ayırma, aynı zamanda boyutsal azaltmanın yanı 
#sıra kümelemeyi gerçekleştiren denetimli bir öğrenme tekniğidir. Konu modellemesi 
#yapmak için TF-IDF şeması ile birlikte kullanılabilir. 
#kaynak:https://stackabuse.com/python-for-nlp-topic-modeling/
import pandas as pd
import numpy as np

reviews_datasets = pd.read_csv(r'Reviews.csv')
reviews_datasets = reviews_datasets.head(20000)
reviews_datasets.dropna()

#TFIDF ile bir belge terim matrisi oluşturacağız.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = tfidf_vect.fit_transform(reviews_datasets['Text'].values.astype('U'))

#Tüm konular için sözcük dağarcığındaki tüm kelimelerin olasılıklarını içeren bir 
#olasılık matrisi oluşturabiliriz. Bunu yapmak için NMFsınıfı sklearn.decompositionmodülden kullanabiliriz.

from sklearn.decomposition import NMF

nmf = NMF(n_components=5, random_state=42)
nmf.fit(doc_term_matrix )

#rastgele kelime dağarcığımızdan 10 kelime alalım
import random

for i in range(10):
    random_id = random.randint(0,len(tfidf_vect.get_feature_names()))
    print(tfidf_vect.get_feature_names()[random_id])
    
#İlk konu için kelimelerin olasılık vektörünü alacağız ve en yüksek olasılıklı 
#on kelimenin dizinlerini alacağız:
first_topic = nmf.components_[0]
top_topic_words = first_topic.argsort()[-10:]

#Bu dizinler artık tfidf_vectgerçek kelimeleri almak için nesneye aktarılabilir .
for i in top_topic_words:
    print(tfidf_vect.get_feature_names()[i])
    
#Şimdi her bir konu için en yüksek olasılıklı on kelimeyi yazalım
for i,topic in enumerate(nmf.components_):
    print(f'Top 10 words for topic #{i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

#Aşağıdaki komut dosyası konuları veri kümesine ekler ve ilk beş satırı görüntüler:
topic_values = nmf.transform(doc_term_matrix)
reviews_datasets['Topic'] = topic_values.argmax(axis=1)
reviews_datasets.head()