# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:38:48 2021

@author: akmal.nordi
"""

import pickle
import streamlit as st
import numpy as np
from PIL import Image

#Start of the app
c1, c2, c3 = st.beta_columns([1,4,1])
with c1:
    image = Image.open('Final Logo DTSF.png')
    st.image(image)

with c2:
    st.title('Concrete Properties Predictor')

with c3:
    image = Image.open('CIMA Logo.png')
    st.image(image)


features = ['Cement', 'Slag', 'Fly Ash', 'Water', 'Super Plasticizer, SP', 'Coarse Aggregate', 'Fine Aggregate']

data = []

values = [273.0, 82.0, 105.0, 210.0, 9.0, 904.0, 680.0]

max_values = [500.0, 300.0, 400.0, 400.0, 50.0, 1200.0, 1200.0]

with st.form(key = 'concrete_form'):
    for i,x in enumerate(features):
        #data.append(st.number_input('%s (kg)' %(x),key=str(i)))
        data.append(st.slider(label='%s (kg)' %(x), min_value=0.0, max_value=max_values[i], value=values[i], step=0.1, key=str(i)))

    submit_button = st.form_submit_button(label='Submit')

data = np.array(data).reshape(1, -1)


# target = ['SLUMP (cm)', 'FLOW (cm)', '28-day Compressive Strength (Mpa)']

# st.header('Prediction 1')
# for i, x in enumerate(target):
#     st.subheader('%s: %d' %(x,pred[:,i]))



# filename = 'model_ANN.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# pred = loaded_model.predict(data)
# st.header('Prediction 2')
# for i, x in enumerate(target):
#     st.subheader('%s: %d' %(x,pred[:,i]))



st.header('Prediction')

filename = 'model_voting_slump.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred1 = loaded_model.predict(data)
st.subheader('Slump: %d cm' %(pred1))

filename = 'model_voting_flow.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred2 = loaded_model.predict(data)
st.subheader('Flow: %d cm' %(pred2))

filename = 'model_voting_strength.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred3 = loaded_model.predict(data)
st.subheader('28-days Compressive Strength: %d MPa' %(pred3))
