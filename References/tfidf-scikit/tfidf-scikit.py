#!/usr/bin/env python
# coding: utf-8

# In[19]:


# IMPORT LIBRARY OR PLUGIN
from scipy.sparse import find
from sklearn.feature_extraction.text import TfidfVectorizer

# DATA SET FROM CSV OR OTHER SOURCES
list_artikel = [
    "saya suka bermain sepak bola",
    "bola adalah teman",
    "bola dunia itu bulat"
]
print("List Article :\n", list_artikel, "\n")


# CONVERT DATA SET INTO TFIDF -> CALCULATE -> COLLECT UNIQUE WORDS
vectorizer = TfidfVectorizer()
vektor = vectorizer.fit_transform(list_artikel)
print("List Feature Names :\n", vectorizer.get_feature_names(), "\n")

# CONVERT VECTOR INTO ARRAY
print("Vector to Array :\n", vektor.toarray(), "\n")


# RESULT BY WORD 
for indexArtikel, v in enumerate(list_artikel):
    print ("Article ke", indexArtikel, ":", list_artikel[indexArtikel])
    for indexKata, kata in enumerate(vectorizer.get_feature_names()):
        print("   ", indexKata, ")", kata, vektor.toarray()[indexArtikel][indexKata])
    print()


# In[ ]:





# In[ ]:




