import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from pickle import dump
from pickle import load
import pickle

# Load the data
df = pd.read_csv("iris.csv")
x = df.drop("species", axis=1)

# Load the trained model
model = load(open('d.sav', 'rb'))

def predict(input):
    input_array = np.asarray(input)
    input_reshape = input_array.reshape(-1, 1)
    prediction = model.predict(input_reshape)
    print(prediction)

# Streamlit app
st.title('Your Machine Learning App')

# Form for user input
sepal_length = st.number_input('Sepal Length')
sepal_width = st.number_input('Sepal Width')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal Width')

# Make a prediction
input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
prediction = model.predict(input_data)
le = load(open('l_e.sav', 'rb'))
pre = le.inverse_transform([prediction])

submit = st.button("Predict")
if submit:
    st.subheader('Predicted Result')
    st.write(pre[0])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")