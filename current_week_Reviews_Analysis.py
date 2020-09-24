# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:53:26 2020

@author: DELL
"""
import re
import datetime
today = datetime.date.today()
print(today)
weekdays = ([today + datetime.timedelta(days=i) for i in range(0 - today.weekday(), 7 - today.weekday())])
weekdays
from datetime import datetime
import EDA_ModelBuilding_SamsungSDcard_AmazonReviews

                    
                    #!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

import os
os.getcwd()
os.chdir("C:\\Users\\DELL\\P30_Group5_ExcelR_Project")

# In[2]:


#df_weekly = pd.read_csv("C:\\Users\\DELL\\P30_Group5_ExcelR_Project\\Samsung_SDcardReviews_latest.csv")
df_weekly = pd.read_csv("C:\\Users\\DELL\\P30_Group5_ExcelR_Project\\Samsung_SDcardReviews_CURRENT_WEEK.csv")
df_weekly = df_weekly.drop_duplicates()

df_weekly['rating'] = df_weekly['stars'].apply(lambda x: re.search(r'\d+',x).group(0))
df_weekly['rating'] = pd.to_numeric(df_weekly['rating'], errors='coerce')
df_weekly = df_weekly.rename(columns ={"comment": "text"})

#df = df.rename(columns ={"comment": "text", "stars": "rating"})
df_weekly.info()

df_weekly['review_date'] = df_weekly['review_date'].apply(lambda x: str.replace(x, 'Reviewed in India on ', ''))
df_weekly['review_date'] = df_weekly['review_date'].apply(lambda x: datetime.strptime(x, '%d %B %Y'))


    
df_weekly = df_weekly[df_weekly['review_date'].apply(lambda x: x in weekdays)] 
df_weekly.info()
 


# In[2]:
# In[3]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 


# In[4]:


import codecs
import unidecode
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def spacy_cleaner(text):
    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)
    apostrophe_handled = re.sub("’", "'", decoded)
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


# In[ ]:


df_weekly['clean_text'] = [spacy_cleaner(t) for t in df_weekly.text]
Word_data_currentweek=df_weekly['clean_text'].astype('str').tolist()
generate_wordcloud(Word_data_currentweek, "word cloud for current week reviews")

# In[ ]:
'''
# word cloud for entire reviews
df_weekly_updated = df_weekly_updated.rename(columns ={"comment": "text"})
df_weekly_updated['clean_text'] = [spacy_cleaner(t) for t in df_weekly_updated.text]

Word_data_updated=df_weekly_updated['clean_text'].astype('str').tolist()
generate_wordcloud(Word_data_updated, "word cloud for reviews inculding current week reviews")
'''
# In[ ]:

# In[ ]:
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
from sklearn.feature_extraction.text import CountVectorizer
#Load it later
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
X_TEST = loaded_vec.fit_transform(df_weekly.clean_text).toarray()
X_TEST_tfidf = tfidfconverter.fit_transform(X_TEST).toarray()

# In[ ]:
#Load it later
#file_name = pickle.load(open("nlp_model.pkl", "rb"))
# In[ ]:
#df_weekly['sentiment'] = SVM.predict(X_TEST_tfidf, file_name)
df_weekly['sentiment'] = SVM.predict(X_TEST_tfidf)
df_weekly.head(10)
# summarize the fit of the model
df_weekly = df_weekly.to_csv('CleanData_Samsung_SDcardReviews_current_week.csv', index = True, encoding = 'utf-8') 

# In[ ]:

'''
df_weekly_updated = pd.read_csv("C:\\Users\\DELL\\P30_Group5_ExcelR_Project\\Samsung_SDcardReviews_weekly_update.csv")
df_current_week = pd.read_csv("C:\\Users\\DELL\\P30_Group5_ExcelR_Project\\Samsung_SDcardReviews_CURRENT_WEEK.csv")
df_current_week = df_current_week[df_current_week['review_date'].apply(lambda x: x in weekdays)] 
df_weekly_updated.append(df_current_week, ignore_index = True)
df_weekly_updated = df_weekly_updated.drop_duplicates()
df_weekly_updated.info()
os.remove('Samsung_SDcardReviews_CURRENT_WEEK.csv')
os.remove('Samsung_SDcardReviews_weekly_update.csv')
# saving the DataFrame as a CSV file 
df_weekly_updated = df.to_csv('Samsung_SDcardReviews_weekly_update.csv', index = True, encoding = 'utf-8') 
'''

