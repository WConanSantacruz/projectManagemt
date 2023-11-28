from datetime import datetime
import glob
import numpy as np
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
import os
import pandas as pd
import shutil
import streamlit as st
import sys
import trimesh
from dotenv import load_dotenv

load_dotenv()

tempCarpet = os.getenv('carpetaTemporal')
scaledFolder=os.getenv('scaledFolder')
savingFolder=f"{tempCarpet}\{scaledFolder}"

def cleanOldFiles():
    # Clearing files, if exist
    if not os.path.exists(tempCarpet):
        os.mkdir(tempCarpet)

    print("DeletingFiles")
    # Delete files in tempDir
    file_list = glob.glob(os.path.join(tempCarpet, '*'))
    for file_path in file_list:
        try:
            if os.path.isfile(file_path):
                print(f"Deleting file: {file_path}")
                os.remove(file_path)
            elif os.path.isdir(file_path):
                print(f"Deleting directory: {file_path}")
                shutil.rmtree(file_path)
            else:
                print(f"Not a file or directory: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    files = os.listdir()
    for file in files:
        if file.endswith(".zip"):
            os.remove(file)

def MainApp():
    fecha_actual = datetime.now()
    current_month=str(fecha_actual.month)
    current_hour=str(fecha_actual.hour)
    current_minute=str(fecha_actual.minute)
    if 'Estado' not in st.session_state:
        st.session_state['Estado'] = 0
    if st.session_state['Estado'] == 0:
        cleanOldFiles()
        st.session_state['Estado']=1
        st.rerun() 
    else:
        st.title("Escalador")
        st.markdown('---')
        name2Quota = f"C{current_month}_{current_hour}_{current_minute}"
        st.markdown(f"Folio: {name2Quota}")
        if(st.session_state['Estado']==2):
            # Add title and subtitle to the main interface of the app
            st.write("Se han escalado los archivos")
            if(st.button("Escalar nuevos archivos")):
                st.session_state['Estado']=0
                st.rerun() 
            return
        elif not(st.session_state['Estado']==1):
            st.write("Error")
            return
        else:
            scale3D=st.number_input('Escalado en porcentaje 1 es el 1% y 100% es la escala normal',1,300,100)
            formats=[]
            for i in trimesh.available_formats():
                formats.append(f".{i}")

            data = st.file_uploader(f"Inserta los archivos para la escalar, solo {formats}", type=formats, accept_multiple_files=True)
            st.markdown('---')
            if len(data) > 0:
                # Converting files if requested
                if st.button('Analiza los archivos'):
                    cleanOldFiles()
                    # Escritura de archivos en tempCarpet
                    for cad in data:
                        source = os.path.join(tempCarpet, cad.name)
                        dest = os.path.splitext(source)[0] + '.stl'
                        with open(source, "wb") as f:
                            f.write(cad.getbuffer())
                        
                        st.write(f'Cargando: {os.path.basename(source)}...')
                    
                    st.write('Analizando, ...')
                    command2Send = f'{sys.executable} scale3D.py {scale3D} '
                    print(command2Send)
                    os.system(command2Send)
                    
                    if os.path.exists(savingFolder):
                        st.markdown('Se han escalado los archivos correctamente')
                    else:
                        st.markdown('Ha habido un error')

                if os.path.exists(savingFolder):
                    if st.button('Preparar Zip'):
                        st.markdown(f'Generando...')

                        shutil.make_archive(name2Quota, 'zip',savingFolder)
                        st.markdown(f'Generado a las {datetime.now()}')
                        zipName=f'{name2Quota}.zip'
                        if os.path.exists(zipName):
                            with open(zipName, 'rb') as f:
                                st.download_button('Descargar', f, file_name=zipName)
                else:
                    st.markdown('Aun no se han escalado los archivos')
            else:
                st.markdown('No se han cargado archivos')
MainApp()