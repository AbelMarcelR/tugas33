
import pickle
import numpy as np
import streamlit as st

st.title("Heart Attack Prediction")
model = pickle.load(open("heartattack123.pkl", "wb"))
age = st.number_input("age")
sex = st.number_input("sex")
cp = st.number_input("cp")
trtbps = st.number_input("trtbps")
chol = st.number_input("chol")
fbs = st.number_input("fbs")
restecg = st.number_input("restecg")
thalachh = st.number_input("Pet")
xng = st.number_input("xng")
oldpeak = st.number_input("Oldpeak")
slp = st.number_input("Slp")
caa = st.number_input("caa")
thall = st.number_input("thall")

btn = st.button("predict")
if btn:
 pred = model.predict(np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,xng,oldpeak,slp,caa,thall]).reshape(1,-1))
 st.write(f"Heart Attack Prediction is(1 is yes , 0 is no): {pred}" )
