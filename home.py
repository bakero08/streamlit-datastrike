import pandas as pd
import streamlit as st
from PIL import Image

def app():
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://e0.365dm.com/21/07/2048x1152/skysports-wsl-womens-super-league_5452968.png");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    st.title("DATASTRIKE! - Shot Analysis Tool")
    #st.info("          Choose the level of analysis from side navigation bar")

#st.markdown("<h1 style='text-align: center; margin-center: 15px;'>Datastrike - Football Analytics Tool</h1>", unsafe_allow_html=True)
#st.markdown("<style> .css-18c15ts {padding-top: 1rem; margin-top:-75px;} </style>", unsafe_allow_html=True)

#image = Image.open('logo.jpg')
# col1, col2 = st.columns(2)


        
