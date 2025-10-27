from flask import Flask, render_template, request, redirect , url_for
import pymysql

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    msg =''
    output = ""    
    if request.method == 'POST':
        review = request.form['review']
        import pandas as pd
        import numpy as np
        df= pd.read_csv('IMDB Dataset.csv', nrows=1000)
        print(df.head())
        df['review'][0]
        df.info()
        df['sentiment'].replace({'positive':1,'negative':0},inplace=True)
        df.head()
        import re
        print(df.iloc[2].review)
        clean=re.compile('<.*?>')
        re.sub(clean,'',df.iloc[2].review)
        def clean_html(text):
            clean=re.compile('<*?.>')
            return re.sub(clean,'',text)
        df['review']=df['review'].apply(clean_html)
        def convert_lower(text):
            return text.lower()
        df['review']=df['review'].apply(convert_lower)
        def remove_special(text):
            x=''
            for i  in text:
                if i.isalnum():
                    x=x+i
                else:
                    x=x+''
            return x
        remove_special('Th%e @ classic use of the word. it is called oz as that is the nickname given to the oswald maximum security state')
        df['review']=df['review'].apply(remove_special)
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stopwords.words('english')
        def remove_stopwords(text):
            x=[]
            for i in text.split():
                if i not in stopwords.words('english'):
                    x.append(i)
            y=x[:]
            x.clear()
            return y
        df['review']=df['review'].apply(remove_stopwords)
        y=[]
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        def stem_words(text):
            for i in text:
                y.append(ps.stem(i))
            z=y[:]
            y.clear()
            return z
        stem_words(['I','loved','loving','it'])
        def join_back(list_input):
            return " ".join(list_input)
        df['review']=df['review'].apply(join_back)
        X=df.iloc[:,0:1].values
        X.shape
        from sklearn.feature_extraction.text import CountVectorizer
        cv=CountVectorizer(max_features=100)
        X=cv.fit_transform(df['review']).toarray()
        X.shape
        y=df.iloc[:,-1].values
        X[0]
        X[0].max()
        X[0].mean()
        y=df.iloc[:,-1].values
        from sklearn.model_selection import train_test_split
        X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2)
        X_train.shape
        X_test.shape
        y_train.shape
        y_test.shape
        from sklearn.naive_bayes import GaussianNB, MultinomialNB
        clf1=GaussianNB()
        clf2=MultinomialNB()
        clf1.fit(X_train,y_train)
        clf2.fit(X_train,y_train)
        y_pred1=clf1.predict(X_test)
        y_pred2=clf2.predict(X_test)
        y_test.shape
        y_pred1.shape
        from sklearn.metrics import accuracy_score
        print("Gaussian",accuracy_score(y_test,y_pred1))
        print("Multinomial",accuracy_score(y_test,y_pred2))
        from textblob import TextBlob
        TextBlob("this movie will not create a revolution")
        a = [[review]] #[["It's just over 2 years since I was diagnosed with #anxiety and #depression. Today I'm taking a moment to reflect on how far I've come since"]]
        a = pd.DataFrame(a)
        print(a)
        X=cv.fit_transform(a[0]).toarray()
        print(X.shape)
        r,c= X.shape
        clf2.fit(X_train[:,:c],y_train)
        y_pred3=clf2.predict(X)
        print("pred ==========",y_pred3)
        if y_pred3[0]==0:
            output = "negative"
        else:
            output= "Positive"
        return render_template('request.html', output=output, Title="request.html")
    return render_template('request.html', output=output, Title="request.html")
if __name__ == "__main__":
    app.run(port=5000,debug=True)