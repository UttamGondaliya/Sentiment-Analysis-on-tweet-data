import streamlit as st
import pickle


import re
import numpy as np
import matplotlib.pyplot as plt
st.title('Sentiment Analyzer')





data_load=st.text('Initialising........')
with open('CountVect.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('LogisticReg.pkl', 'rb') as fo:
    model = pickle.load(fo)

data_load.text('Initialising........Done!')


st.subheader('In given text box below, please enter a text/tweet or post which you want to Evaluate.')
st.subheader('Model will evaluate your text and will return sentiment behind that text')

text=st.text_input('Enter your text')


def decontracted(phrase):
   # specific
   phrase = re.sub(r"won\'t", "will not", phrase)
   phrase = re.sub(r"can\'t", "can not", phrase)
    
            # general
   phrase = re.sub(r"n\'t", " not", phrase)
   phrase = re.sub(r"\'re", " are", phrase)
   phrase = re.sub(r"\'s", " is", phrase)
   phrase = re.sub(r"\'d", " would", phrase)
   phrase = re.sub(r"\'ll", " will", phrase)
   phrase = re.sub(r"\'t", " not", phrase)
   phrase = re.sub(r"\'ve", " have", phrase)
   phrase = re.sub(r"\'m", " am", phrase)
   return phrase
def preprocess(txt):
   txt=txt.replace('&amp;','and')
   txt=re.sub(r'http\S+', '', txt)
   txt=decontracted(txt)
   txt=" ".join(filter(lambda x:x[0]!='@', txt.split()))
   txt=" ".join(filter(lambda x:x[0]!='#', txt.split()))
   txt=" ".join(filter(lambda x:x[-4:]!='.com', txt.split()))
   txt=re.sub('[^A-Za-z ]+','', txt)
   return txt.lower().strip()
text=decontracted(text)
text=preprocess(text)

x=cv.transform([text]).todense()
#st.dataframe(x)

pred=model.predict_proba(x)

i=np.argmax(pred)

if st.checkbox("Check Result"):
    if i==0:
        st.header('Given Sentence is Negative')
    if i==1:
        st.header('Given Sentence is Neutral')
    if i==2:
        st.header('Given Sentence is Positive')
    
    #st.subheader(pred)
    sizes = pred[0]
    labels = 'Negative', 'Neutral', 'Positive'
    
    explode = (0.1, 0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    st.pyplot(fig1)
