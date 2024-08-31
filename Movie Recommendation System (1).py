#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


movies=pd.read_csv('dataset.csv')


# In[3]:


movies.head(10)


# In[4]:


movies.head(10)


# In[5]:


movies.info()


# In[6]:


movies.isnull().sum()


# In[7]:


movies.columns


# In[8]:


movies=movies[['id', 'title', 'overview', 'genre']]


# In[9]:


movies


# In[10]:


movies['tags'] = movies['overview']+movies['genre']


# In[11]:


movies


# In[12]:


new_data  = movies.drop(columns=['overview', 'genre'])


# In[13]:


new_data


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer


# In[15]:


cv=CountVectorizer(max_features=10000, stop_words='english')


# In[16]:


cv


# In[17]:


vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()


# In[18]:


vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()


# In[19]:


vector.shape


# In[20]:


from sklearn.metrics.pairwise import cosine_similarity


# In[21]:


similarity=cosine_similarity(vector)


# In[22]:


similarity


# In[23]:


new_data[new_data['title']=="The Godfather"].index[0]


# In[24]:


distance = sorted(list(enumerate(similarity[2])), reverse=True, key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].title)


# In[25]:


def recommand(movies):
    index=new_data[new_data['title']==movies].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)


# In[26]:


recommand("Iron Man")


# In[27]:


import pickle


# In[28]:


pickle.dump(new_data, open('movies_list.pkl', 'wb'))


# In[29]:


pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[ ]:





# In[30]:


pickle.load(open('movies_list.pkl', 'rb'))


# In[ ]:




