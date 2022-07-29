# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:27:37 2022

@author: hp
"""
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Title and sidebar
st.title("Model Deployment: Gold Price Prediction")
st.sidebar.header("User input parameters")


# Input function
def user_input_features():
    fc_days = st.sidebar.number_input("Insert number of days to predict.")
    return int(fc_days)


prediction_days = user_input_features() 


# Loading trained model and making predictions
loaded_model = pickle.load(open("C:\\Users\\hp\\Downloads\\model_deploy\\best_model.pickle", "rb"))
predictions = loaded_model.forecast(652 + prediction_days)


# Output predictions
st.subheader('Predicted Result')
st.write(predictions[652:])
