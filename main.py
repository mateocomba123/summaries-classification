# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:37:28 2021

@author: Mateio
"""

import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

#Stopwords corpus
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

#Preprocessing function
numbers = ['0','1','2','3','4','5','6','7','8','9']
def prepare_text(text):
    tokens = text
    tokens = [token for token in tokens if len(token) > 2] #Remove words with less than 2 characters
    tokens = [token for token in tokens if token not in en_stop] #Remove stopwords
    tokens = [token for token in tokens if not any(number in token for number in numbers)]  #Remove numbers   
    tokens = [token.replace('[^\w\s]','') for token in tokens] #Remove punctuation
    tokens = [token.replace('_','') for token in tokens] #Remove punctuation
    return tokens

#tokens to str
def tostr(text):
    str1 = "" 
    for word in text: 
        str1 += word.lower()
        str1 += ' '
    return str1

import streamlit as st
from streamlit import components

st.set_page_config(layout="wide")
st.title('Model playground')

st.write("""
# Input some summary to make a prediction
""")

sentence = st.text_area('Input your sentence here:')



if sentence != '':
    #Load model and vectorizer
    V_Cat = pickle.load(open('Vectorizer','rb'))
    M_Cat = pickle.load(open('Model','rb'))
    
    #Preprocess the sentence
    Test = word_tokenize(sentence)
    Test = prepare_text(Test)
    Test = tostr(Test)
    TestExplainer = Test
    
    ####Category
    Test2 = V_Cat.transform([Test])
    predict = (M_Cat.predict(Test2))
    predict = predict[0]
    
    cm_labels = ['Access Management', 'Application', 'Cloud', 'Data Center',
       'Deskside', 'Employee Status', 'Network', 'Operations',
       'Process Control', 'Radio', 'Security', 'Voice Communication']
    prob = M_Cat.predict_proba(Test2)
    probas = pd.DataFrame()
    probas['Category'] = cm_labels
    probas['Prob'] = prob[0]
    print(predict)
    
    types = ['Access Management', 'Application', 'Cloud', 'Data Center',
       'Deskside', 'Employee Status', 'Network', 'Operations',
       'Process Control', 'Radio', 'Security', 'Voice Communication']
    
    #Delete spaces for path string
    tipo = ''
    if predict == 'Access Management':
        tipo = 'AccessManagement'
    if predict == 'Deskside':
        tipo = 'Deskside'
        
    if predict == 'Access Management' or predict == 'Deskside':
        
        #Load Type model and vectorizer
        V_Type = pickle.load(open("VectorizerType" + tipo,'rb'))
        M_Type = pickle.load(open("ModelType" + tipo,'rb'))
       
        ####Type
        Test2 = V_Type.transform([Test])
        predictType = (M_Type.predict(Test2))
        predictType = predictType[0]
        
        types = np.load('labels' + tipo + '.npy',allow_pickle=True)
        TypeProb = M_Type.predict_proba(Test2)
        TypeProbas = pd.DataFrame()
        TypeProbas['Category'] = types
        TypeProbas['Prob'] = TypeProb[0]
        print(predictType)



if sentence != '':
    
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(V_Cat, M_Cat)
    
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['Access Management', 'Application', 'Cloud', 'Data Center',
       'Deskside', 'Employee Status', 'Network', 'Operations',
       'Process Control', 'Radio', 'Security', 'Voice Communication'])
    exp = explainer.explain_instance(TestExplainer, c.predict_proba, num_features=10, top_labels=1)
    html_data = exp.as_html()

with st.sidebar:
    if sentence != '':
        st.dataframe(probas.sort_values('Prob', ascending=False).style.highlight_max(color = 'lightgreen', axis = 0))
        
        if predict == 'Access Management' or predict == 'Deskside':
             st.dataframe(TypeProbas.sort_values('Prob', ascending=False).style.highlight_max(color = 'lightgreen', axis = 0))
        else:
            st.write('')       
    else:
        st.write('')

if sentence != '':
            
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category precited: " + predict)      
        Category = st.selectbox('Select Category', probas.sort_values('Prob', ascending=False)['Category'], key="box1")
    
    if predict == 'Access Management' or predict == 'Deskside':
        with col2:
            st.subheader("Type predicted: " + predictType)
            Type = st.selectbox('Select Type', TypeProbas.sort_values('Prob', ascending=False)['Category'], key="box2")
        
        
        c2 = make_pipeline(V_Type, M_Type)
        explainer2 = LimeTextExplainer(class_names=types)
        exp2 = explainer2.explain_instance(TestExplainer, c2.predict_proba, num_features=10, top_labels=1)
        html_data2 = exp2.as_html()
        components.v1.html(html_data2,width=1000, height=300, scrolling=True)
        #Showing explainability charts    
        components.v1.html(html_data,width=1000, height=300, scrolling=True)
    
    
    
    
