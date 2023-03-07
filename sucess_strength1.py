# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:17:04 2023

@author: krishna
"""
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import numpy.ma as ma
from io import BytesIO
import requests
linkn = 'https://github.com/anandaiml19/End-to-End-Machine-Learning-Project-with-Deployment-using-Streamlit-FastAPI-and-AWS-EC2/blob/main/strengthhh1_concrete.sav?raw=true'
nfile = BytesIO(requests.get(linkn).content)

# load the model
load_strength = pickle.load(nfile)

# create a prediction function
def strength_concrete(concc):
    
    # convert to numpy array
    output_numpyy = np.asarray(concc)

    # reshape the data
    reshape_strength = output_numpyy.reshape(1,-1) 

    # model predction
    output_strength = load_strength.predict(reshape_strength)
    masked_removed = ma.masked_array( output_strength, mask = [False])
    strength_float = masked_removed.__float__()
    return "The strength of concrete is:", round(strength_float,2)


def main():
    
    #image input
    col1,col2,col3 = st.columns(3)
    
    with col1:
        st.write(' ')
        
    with col2:
        image1 = Image.open('D:/EDA_LR/Capstone_Project_Cement/concrete.jpg')
        image1 = image1.resize((220,160))
        st.image(image1,use_column_width=False)
        
    with col3:
        st.write(' ')
    
    # title of app
    st.title("Concrete Strength Predction Web App")
    
    # get input data from user
    
    cement = st.number_input('Cement Component Value')
    slag = st.number_input('Blast Furnance Slag Value')
    ash = st.number_input('Fly Ash Value')
    water = st.number_input('Water Mixed Value')
    superplastic = st.number_input('Super Plasticizer Mixed Value')
    coarseagg = st.number_input('Coarse Aggregate Mixed Value')
    fineagg = st.number_input('Fine Aggregate Mixed Value')
    age = st.number_input('No of Days Dried')
    
    
    # plain final result variable
    strengthc = ''
    
    if st.button('Predict Concrete Strength'):
        strengthc = strength_concrete([cement, slag, ash, water, superplastic, coarseagg, fineagg, age])
        
    st.success(strengthc)
    
    
if __name__ == '__main__':
    main()
        

    