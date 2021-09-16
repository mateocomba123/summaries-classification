
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:37:28 2021

@author: Mateio
"""

import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import numpy as np

import string
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def prepare_text(text):
    tokens = text
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token.isnumeric()==False]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [token for token in tokens if '0' not in token]
    tokens = [token for token in tokens if '1' not in token]
    tokens = [token for token in tokens if '2' not in token]
    tokens = [token for token in tokens if '3' not in token]
    tokens = [token for token in tokens if '4' not in token]
    tokens = [token for token in tokens if '5' not in token]
    tokens = [token for token in tokens if '6' not in token]
    tokens = [token for token in tokens if '7' not in token]
    tokens = [token for token in tokens if '8' not in token]
    tokens = [token for token in tokens if '9' not in token]
    return tokens

def tostr(text):
    str1 = "" 
    for word in text: 
        str1 += word
        str1 += ' '
    return str1

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

import streamlit as st
from streamlit import components

st.set_page_config(layout="wide")
st.title('Model playground')

st.write("""
# Input some summary to make a prediction
""")

sentence = st.text_area('Input your sentence here:')


#Loading models and vectorizers
vectorizer = pickle.load(open('Vectorizer','rb'))
Log_Ridge = pickle.load(open('Model','rb'))

vectorizer2 = pickle.load(open('Vectorizer2','rb'))
Log_Ridge2 = pickle.load(open('Model2','rb'))


#preprocessing
Test = remove_punctuations(sentence)
Test = word_tokenize(Test)
Test = prepare_text(Test)
Test = tostr(Test).lower()
TestExplainer = Test
Test2 = Test

####Category
Test = vectorizer.transform([Test])
predict = (Log_Ridge.predict(Test))
predict = predict[0]

cm_labels = ['Access Management', 'Application', 'Cloud', 'Data Center',
   'Deskside', 'Employee Status', 'Network', 'Operations',
   'Process Control', 'Radio', 'Security', 'Voice Communication']
prob = Log_Ridge.predict_proba(Test)
probas = pd.DataFrame()
probas['Category'] = cm_labels
probas['Prob'] = prob[0]


####Type - Access Management
Test2 = vectorizer2.transform([Test2])
predict2 = (Log_Ridge2.predict(Test2))
predict2 = predict2[0]

cm_labels2 = pd.read_csv('cm_labels.csv')
cm_labels2 = np.unique(cm_labels2).tolist()
prob2 = Log_Ridge2.predict_proba(Test2)
probas2 = pd.DataFrame()
probas2['Category'] = cm_labels2
probas2['Prob'] = prob2[0]


if sentence != '':
    
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(vectorizer, Log_Ridge)
    
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['Access Management', 'Application', 'Cloud', 'Data Center',
       'Deskside', 'Employee Status', 'Network', 'Operations',
       'Process Control', 'Radio', 'Security', 'Voice Communication'])
    exp = explainer.explain_instance(TestExplainer, c.predict_proba, num_features=10, top_labels=1)
    html_data = exp.as_html()

with st.sidebar:
    if sentence != '':
        st.dataframe(probas.sort_values('Prob', ascending=False).style.highlight_max(color = 'lightgreen', axis = 0))
        
        if predict == 'Access Management':
             st.dataframe(probas2.sort_values('Prob', ascending=False).style.highlight_max(color = 'lightgreen', axis = 0))
        else:
            st.write('')       
    else:
        st.write('')

if sentence != '':
            
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category precited: " + predict)      
        Category = st.selectbox('Select Category', probas.sort_values('Prob', ascending=False)['Category'], key="box1")
    
    if predict == 'Access Management':
        with col2:
            st.subheader("Type predicted: " + predict2)
            Type = st.selectbox('Select Type', probas2.sort_values('Prob', ascending=False)['Category'], key="box2")
        
        
        c2 = make_pipeline(vectorizer2, Log_Ridge2)
        explainer2 = LimeTextExplainer(class_names=cm_labels2)
        exp2 = explainer2.explain_instance(TestExplainer, c2.predict_proba, num_features=10, top_labels=1)
        html_data2 = exp2.as_html()
        components.v1.html(html_data2,width=1000, height=300, scrolling=True)
    #Showing explainability charts    
    components.v1.html(html_data,width=1000, height=300, scrolling=True)
    
    
