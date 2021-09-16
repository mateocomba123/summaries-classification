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

vectorizer = pickle.load(open('C:/Users/Mateio/Desktop/freeport/Vectorizer','rb'))
Log_Ridge = pickle.load(open('C:/Users/Mateio/Desktop/freeport/Model','rb'))

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



st.set_page_config(layout="wide")
st.title('Model playground')

import streamlit as st
from streamlit import components
import eli5



st.write("""
# Input some summary to make a prediction
""")



sentence = st.text_area('Input your sentence here:')

#Test = "asdasd asd"
Test = remove_punctuations(sentence)
Test = word_tokenize(Test)
Test = prepare_text(Test)
Test = tostr(Test).lower()
TestExplainer = Test
Test = vectorizer.transform([Test])
predict = (Log_Ridge.predict(Test))
predict = predict[0]


if sentence != '':
    st.subheader('The predicted value is: ' + predict)
    
    from lime import lime_text
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(vectorizer, Log_Ridge)
    
    
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['Access Management', 'Application', 'Cloud', 'Data Center',
       'Deskside', 'Employee Status', 'Network', 'Operations',
       'Process Control', 'Radio', 'Security', 'Voice Communication'])
    exp = explainer.explain_instance(TestExplainer, c.predict_proba, num_features=10, top_labels=1)
    html_data = exp.as_html()
    
    components.v1.html(html_data,width=1000, height=300, scrolling=True)
    
    html_object = eli5.show_weights(Log_Ridge, vec=vectorizer, top=20)

    raw_html = html_object._repr_html_()

    components.v1.html(raw_html,width=1200, height=600, scrolling=True)


with st.sidebar:
    if sentence != '':
        cm_labels = pd.read_csv('C:/Users/Mateio/Desktop/freeport/cm_labels.csv')
        cm_labels = np.unique(cm_labels)
        prob = Log_Ridge.predict_proba(Test)
        probas = pd.DataFrame()
        probas['Category'] = cm_labels
        probas['Prob'] = prob[0]
        
        st.dataframe(probas.sort_values('Prob', ascending=False).style.highlight_max(color = 'lightgreen', axis = 0))
    else:
        st.write('')
    
    