#!/usr/bin/env python
# coding: utf-8

# # Final Step - ML Model

# ### Read in text

# In[3]:


import pandas as pd
import re
import string
import nltk
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']


# ### Create function to remove punctuation, tokenize, remove stopwords, and stem

# In[4]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# ### Apply CountVectorizer

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
print(X_counts.shape)
print(count_vect.get_feature_names())
print(X_counts)


# # Seperating Dependent and Independent Variable

# In[7]:


X = X_counts.toarray()
y = data.iloc[:,0].values
print(y)


# # Encoding The Dependent Variable

# In[9]:


#To deal with categorical data we had to convert it into numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
#converting first column into integer values
y=labelencoder_y.fit_transform(y)
print(y)


# # Splitting The Dataset

# In[10]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Using Naive Bayes Classifier

# In[13]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# # Predicting Results

# In[14]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# # Confusion Matrix

# In[15]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[16]:


cm


# In[17]:


(841+143)/(841+143+118+12)


# In[ ]:




