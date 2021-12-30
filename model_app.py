import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df=pd.read_csv('t_b.csv')
df.drop(columns=df.columns[0],inplace=True)
df=df.fillna('NA')
df=df.loc[df.clean_text!='NA']
df=df.loc[df.target!='NA']



data=df['clean_text'].values
vectorizer = CountVectorizer(stop_words={'english'},ngram_range=(1,2),min_df=300)
cv= vectorizer.fit(data)
train_x=cv.transform(data).todense()


model=LogisticRegression()

model.fit(train_x,df.target.values)


with open('CountVect.pkl', 'wb') as fout:
    pickle.dump(cv, fout)
with open('LogisticReg.pkl', 'wb') as f:
    pickle.dump(model, f)
