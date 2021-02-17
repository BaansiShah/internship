import streamlit as st 
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge

from sklearn.metrics import accuracy_score

import datetime
earlier = datetime.datetime.now()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("app.css")

# Apple Data
googleSheetId = '1zxpJ2BoFnkhYpVph_uq0c_KtbkNNTY6iU4pzEoCzxH8'
worksheetName = 'Apple'
URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
	googleSheetId,
	worksheetName
)
df_apple = pd.read_csv(URL)

# Google Data
worksheetName = 'Google'
URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
	googleSheetId,
	worksheetName
)
df_google = pd.read_csv(URL)

# Microsoft Data
worksheetName = 'Microsoft'
URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
	googleSheetId,
	worksheetName
)
df_micro = pd.read_csv(URL)

st.title('Analysis of US Stock Market')

#st.write("""
## Explore different classifier and datasets
#Which one is the best?
#""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Apple', 'Google', 'Microsoft')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Linear Regression', 'SVM', 'BayesianRidge')
)

Start_Date = st.sidebar.date_input('Start Date')
End_Date = st.sidebar.date_input('End Date')

def get_dataset(name):
    data = None
    if name == 'Apple':
        data = df_apple
    elif name == 'Google':
        data = df_google
    else:
        data = df_micro
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    return X, y

X, y = get_dataset(dataset_name)
st.write(f"## Features of the Dataset")
st.write(X)
st.write(f"## Target Value of the Dataset")
st.write(y)

st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Linear Regression':
        L = st.sidebar.slider('L', 1, 10)
        params['L'] = L
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 20.0)
        params['C'] = C
    else:
        n_iter = st.sidebar.slider('N_Iter', 300, 600)
        params['n_iter'] = n_iter
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Linear Regression':
        clf = LinearRegression(n_jobs=params['L'])
    elif clf_name == 'SVM':
        clf = svm.SVR(C=params['C'])
    else:
        clf = BayesianRidge(n_iter=params['n_iter'])
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = clf.score(X_test, y_test)

user_open = float(st.text_input("Enter value of Open price", 2))
user_high = float(st.text_input("Enter value of High price", 2))
user_low = float(st.text_input("Enter value of Low price", 2))
user_volume = int(st.text_input("Enter value of Volume", 2))

def predict():
    pred = clf.predict([[user_open, user_high, user_low, user_volume]])
    pred[0] = round(pred[0], 2)
    st.write(f"## Prediction for above information is : {pred[0]} :smile:")
    return
if st.button('Predict'):
    predict()


st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)

now = datetime.datetime.now()

diff = now - earlier

st.write(f"## Delay in seconds = {diff.seconds} ")