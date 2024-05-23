import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
url = os.getenv('WebAppURL')
temp_dir = "tempUnl"
path2getQuotas = url + os.getenv('quotaURL')
path2download = url + os.getenv('downloadURL')
params = {'key': os.getenv('AccessKey')}


def MainApp():
    st.title("Sistema de Captis")
    st.markdown("Selecciona la pestaña que necesites")
    
MainApp()
    