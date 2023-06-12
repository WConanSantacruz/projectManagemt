import json
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import requests
import wget

load_dotenv()
url = os.getenv('WebAppURL')
temp_dir = "tempUnl"
path2getQuotas = url + os.getenv('quotaURL')
path2download = url + os.getenv('downloadURL')
params = {'key': os.getenv('AccessKey')}


def download_file(url, filename, output_dir):
    wget.download(url, out=os.path.join(output_dir, filename))

def clean_temp_dir(temp_dir):
    file_list = [file for file in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, file))]
    for file_name in file_list:
        file_path = os.path.join(temp_dir, file_name)
        os.remove(file_path)

def MainApp():
    st.title("Bienvenido al administrador de proyectos de CAPTIS")
    response = requests.get(path2getQuotas, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        st.dataframe(df)
        
        selecionada = st.selectbox("Por favor selecciona la cotización a descargar", df["id"], format_func=lambda x: df[df["id"] == x]["id"].values[0])
        selected = df[df["id"] == selecionada]
        mat = selected["idMaterial"].to_json()
        dataMaterial = json.loads(mat)
        keys = list(dataMaterial.keys())
        dataMaterial=dataMaterial[keys[0]]
        
        st.markdown("## ID")
        st.markdown(selected["id"].to_string())
        st.markdown("## Fecha deseada")
        st.markdown(selected["wantedDate"].to_string())
        st.markdown("## Descripcion")
        st.markdown(selected["description"].to_string())
        st.markdown("### Material")
        st.markdown(dataMaterial["name"])
        st.markdown("### Archivos adjuntos")
        lk=json.loads(selected["AttachedFile"].to_json())
        keysInFiles=list(lk.keys())
        infoFiles = lk[keysInFiles[0]]
        
        for data in infoFiles:
            st.markdown(f" {data['name']}")
        

        os.makedirs(temp_dir, exist_ok=True)  # Create the temp directory if it doesn't exist

        if st.button("Descargar archivos"):

            clean_temp_dir(temp_dir)  # Clean the temp directory before downloading new files
            with st.spinner("Descargando archivos..."):
                for i, data in enumerate(infoFiles):
                    st.markdown(f" {data['name']}")
                    st.markdown(data["path"])
                    parDow = {'key': os.getenv('AccessKey'), 'file': data["path"]}
                    url = path2download + '?' + '&'.join([f'{key}={value}' for key, value in parDow.items()])
                    # st.markdown(url)
                    download_file(url, filename=data["name"], output_dir=temp_dir)
                    st.text((i + 1) / len(infoFiles))

        if st.button("Open the folder"):
            os.startfile(temp_dir)
    else:
        st.write(f"Error: {response.status_code}")

MainApp()
    