# CSCI 544 - Applied Natural Language Processing
# CSCI 544 - Assignment 1

# Name: Sri Manvith Vaddeboyina
# USC ID: 1231409457

# 1. Data Preparation

# Importing necessary libraries/packages</h3>

import re
import sys 
import nltk
import pandas as pd
import numpy as np
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline

from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support as score

nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')


# Read Data

# Reading Amazon US Beauty Reviews (tsv) dataset and retaining only the following two columns:
# 1. review_body
# 2. star_rating

df = pd.read_csv('amazon_reviews_us_Beauty_v1_00.tsv', sep='\t',usecols =['star_rating','review_body'])

# Dropping the entire rows where any of the column contains NA value
df.dropna(inplace=True)

# Keep Reviews and Ratings

# Create a three-class classification problem according to the ratings.
# Ratings:
# 1 and 2 - class 1
# 3 - class 2
# 4 and 5 - class 3

df = df[
        df['star_rating'].eq('1') | 
        df['star_rating'].eq('2') | 
        df['star_rating'].eq('3') | 
        df['star_rating'].eq('4') | 
        df['star_rating'].eq('5')
       ]


# Verifying the datatype of each column and setting them correctly
df['star_rating']=df['star_rating'].astype(int)
df['review_body']=df['review_body'].astype(str)


# Creating a 3-class classification on ratings

def condition(x):
    if x==1 or x==2:
        return 1
    elif x==3:
        return 2
    elif x==4 or x==5:
        return 3
    
df['rating'] = df['star_rating'].apply(condition)


# We form three classes and select 20000 reviews randomly from each class.

# Randomly selecting 20000 reviews from each of class 1,2 and 3.
# Total: 60000 reviews
df=df.groupby('rating').sample(n=20000)
df.drop(['star_rating'],inplace=True,axis=1)


# Function to find the average length of reviews

def avg_len_reviews(column_name):
    length=0
    for i in range(len(df)):
        length=length+len(df[column_name].iloc[i])
    avg_length=length/len(df)
    return avg_length





# 2. Data Cleaning
# Removing the following as part of data cleaning:
# 1. URLs
# 2. HTML tags
# 2. Contractions Expansion
# 3. Non-alphabetic characters
# 4. Converting text to lower case
# 5. Removing extra spaces
# 6. Removing emojis

def remove_urls (text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    return(text)

def remove_contractions(text):
    expanded_words = []   
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    return expanded_text

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f" 
        u"\u3030"
        "]+", re.UNICODE)
    return re.sub(emoj, '', data)

remove_non_english = lambda s: re.sub(r'[^a-zA-Z ]', ' ', s)
remove_spaces =  lambda s: re.sub(' +', ' ', s)


def cleaning(text):
        #remove urls
        text=remove_urls(text)
        
        #remove html tags
        text = BeautifulSoup(text, "lxml").text

        #remove contractions
        text=remove_contractions(text)
        
        #remove non-alphabetic chars
        text=remove_non_english(text)
        
        #lowercase
        text=text.lower()
        
        #remove extra spaces
        text=remove_spaces(text)
        
        #remove emojis
        text=remove_emojis(text)

        return text

df['cleaned_text_reviews'] = list(map(cleaning, df.review_body))


# Average length of reviews before cleaning and after cleaning
print(avg_len_reviews("review_body"),avg_len_reviews("cleaned_text_reviews"),sep=", ")





# 3. Preprocessing

# Remove the stop words 

# Removing the stop words in english language

# remove stop words
def stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

df['clean_stop'] = list(map(stop_words, df.cleaned_text_reviews))


# Perform lemmatization

# lemmetization
def lemmatized_words(text):
    lemm = nltk.stem.WordNetLemmatizer()
    lst=nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lemm.lemmatize(w) for w in lst])
    return lemmatized_output

df['lemmetized_data'] = list(map(lemmatized_words, df.clean_stop))


# Average length of reviews before preprocessing and after preprocessing
print(avg_len_reviews("cleaned_text_reviews"),avg_len_reviews("lemmetized_data"),sep=", ")

# Splitting Data (80%-20%)
X=df['cleaned_text_reviews']
y=df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)




# 4. Feature Extraction

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Function to print all metrics - precision, recall and F1-score along with the averages as per the assignment guidelines
def results(y_test,y_pred):
    precision, recall, fscore, support = score(y_test, y_pred)
    averages=score(y_test, y_pred, average='weighted')
    for i in range(len(precision)):
        print(precision[i],recall[i],fscore[i],sep=", ")
    print(averages[0],averages[1],averages[2],sep=", ")


# 5. Perceptron
perceptron_text_clf = Perceptron()
perceptron_text_clf.fit(X_train, y_train)
perceptron_predictions = perceptron_text_clf.predict(X_test)

# Classification results for Perceptron model on test data (20%)
results(y_test,perceptron_predictions)


# 6. SVM
svc_text_clf = LinearSVC()
svc_text_clf.fit(X_train, y_train)
svc_predictions = svc_text_clf.predict(X_test)

# Classification results for SVM (SVC) model on test data (20%)
results(y_test,svc_predictions)


# 7. Logistic Regression
lr_text_clf = LogisticRegression(max_iter=500)
lr_text_clf.fit(X_train, y_train)
lr_predictions = lr_text_clf.predict(X_test)

# Classification results for Logistic Regression model on test data (20%)
results(y_test,lr_predictions)


# 8. Naive Bayes
mnb_text_clf = MultinomialNB()
mnb_text_clf.fit(X_train, y_train)
mnb_predictions = mnb_text_clf.predict(X_test)

# Classification results for Multinomial Naive Bayes model on test data (20%)
results(y_test,mnb_predictions)
