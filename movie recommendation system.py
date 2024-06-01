#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


credits.head(1)['cast'].values


# In[6]:


movies=movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# In[8]:


movies['original_language'].value_counts()


# In[9]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna(inplace=True)


# In[13]:


movies.duplicated().sum()


# In[14]:


movies.iloc[0].genres


# In[15]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','SciFi']


# In[16]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
        


# In[17]:


movies['genres']=movies['genres'].apply(convert)


# In[18]:


movies.head()


# In[19]:


movies['keywords']=movies['keywords'].apply(convert)


# In[20]:


movies.head()


# In[21]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
        
    return L


# In[22]:


movies['cast']=movies['cast'].apply(convert3)


# In[23]:


movies.head()


# In[24]:


movies['crew'][0]
   
   


# In[25]:


def fetch_director(obj):
    L=[]
   
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
        
    return L


# In[26]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[27]:


movies.head()


# In[28]:


movies['overview'][0]


# In[29]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[30]:


movies.head()


# In[31]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[32]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['crew']


# In[33]:


movies.head()


# In[34]:


new_df=movies[['movie_id','title','tags']]


# In[35]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[36]:


new_df.head()


# In[37]:


import nltk


# In[38]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[39]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[40]:


new_df['tags']=new_df['tags'].apply(stem)


# In[41]:


new_df['tags'][0]


# In[42]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[43]:


new_df.head()


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=5000,stop_words='english')
cv.fit_transform(new_df['tags']).toarray()


# In[45]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[46]:


vectors


# In[47]:


vectors[0]


# In[48]:


cv.get_feature_names()


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity


# In[50]:


similarity=cosine_similarity(vectors)


# In[51]:


similarity


# In[52]:


new_df[new_df['title']=='The Lego Movie'].index[0]


# In[53]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    movie_list=sorted(list(enumerate(similarity[movie_index])),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[54]:


recommend('Avatar')


# In[55]:


import pickle


# In[57]:


pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




