#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np   
import pandas as pd  
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 


# Data collection and pre-process

# In[2]:


movies_data=pd.read_csv(r"D:\ML_WORK\movies.csv")


# In[3]:


movies_data.head()


# In[4]:


movies_data.shape


# In[5]:


selected_features=["genres",'keywords','tagline','cast','director']
print(selected_features)


# In[6]:


#replacing null values with null string
for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna('')


# In[7]:


#combining all the features(concating all the features all together)

combined_features=movies_data['genres']+" "+movies_data['keywords']+" "+movies_data['tagline']+" "+movies_data['cast']+" "+movies_data['director']


# In[8]:


combined_features.head


# In[9]:


#creating instance of vectorizer
vectorizer=TfidfVectorizer() #-->loading this TfidVec.. that we have imported from Sklearn into vectorizer


# In[10]:


#now fitting and transforming all the data
feature_vectors=vectorizer.fit_transform(combined_features)


# In[11]:


print(feature_vectors) # now we can fit this data into the ML model this will give numeric value to text data


# In[12]:


#Now finding the similarity score-> using cosine similarity, also imported from sklearn
similarity=cosine_similarity(feature_vectors)


# In[13]:


print(similarity)


# In[14]:


print(similarity[0][:9]) # so basically it will pick 1st movie, and compare with all to give a similarity score, similarly for all movie.


# In[15]:


similarity.shape #1. is movie index, 2nd is similarity score


# In[16]:


# now we can ask user to give input fav movie name, 

movie_name=input("Enter your favourite movie name : ")


# In[17]:


#create a list that contain all the movie names given in that dataset

list_of_all_titles=movies_data['title'].tolist()
print(list_of_all_titles)


# In[18]:


#finding the close match for the movie name given by user

find_close_match= difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)


# In[19]:


# we want only 1 value, which will match exactly, not all, only the most similar

close_match= find_close_match[0]
print(close_match)


# In[20]:


#finding the index of this movie with title, based on title

index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
print(index_of_the_movie)


# In[21]:


#getting list of similar movie->now we will take this index and find the similar movie to this

similarity_score= list(enumerate(similarity[index_of_the_movie]))  #enumerate is used to carry a loop in list

print(similarity_score) # 1st value index of movie, 2nd is similarity score compare to input


# In[22]:


len(similarity_score)


# In[23]:


# we want only which has highest similarity value, we will sort this from high similarity value to lowest

sorted_similar_movies=sorted(similarity_score,key= lambda x:x[1],reverse=True) 


# In[24]:


#we will print the name of similar movie based on index

print('Movies suggested for you :\n')

i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if (i<16):
        print(i,'.',title_from_index)
        i+=1


# In[25]:


# Collecting code at one place


# In[26]:


movie_name=input("Enter your favourite movie name : ")

list_of_all_titles=movies_data['title'].tolist()

find_close_match= difflib.get_close_matches(movie_name,list_of_all_titles)

close_match= find_close_match[0]

index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]

similarity_score= list(enumerate(similarity[index_of_the_movie])) 

sorted_similar_movies=sorted(similarity_score,key= lambda x:x[1],reverse=True) 

print('Movies suggested for you :\n')

i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if (i<16):
        print(i,'.',title_from_index)
        i+=1

