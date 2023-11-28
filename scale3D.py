import os
from datetime import datetime
import numpy as np
import sys
import pyvista as pv
import pandas as pd
import trimesh
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

def getInfoMesh(actualFile, savingFolder, scale, try2Fix=True):
    try:
        mesh = trimesh.exchange.load.load_mesh(actualFile)

        if not mesh.is_watertight and try2Fix:
            print("The STL file is being repaired.")
            if not (trimesh.repair.fill_holes(mesh) or trimesh.repair.fix_winding(mesh)):
                trimesh.repair.stitch(mesh)
            else:
                print('Repaired')

        volume = mesh.mass_properties['volume']
        print("The volume of the mesh is:", volume)

        Sx, Sy, Sz = np.ptp(mesh.vertices, axis=0)
        mesh.vertices -= mesh.center_mass

        # Create a scaling matrix
        scaling_matrix = trimesh.transformations.scale_matrix(scale)

        # Apply scaling directly to mesh vertices
        mesh.apply_transform(scaling_matrix)

        area = mesh.area

        # Get the filename without extension
        file_name = os.path.splitext(os.path.basename(actualFile))[0]

        # Construct the output path with the scaled filename
        output_path = f'{savingFolder}/{file_name}_scaled.stl'
        print(output_path)

        # Export the scaled mesh
        mesh.export(output_path)

        return Sx, Sy, Sz, volume, area

    except Exception as e:
        # Print the error message and traceback
        print(f"Error processing mesh: {e}")
        return 0, 0, 0, 0, 0
    
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        # Directory does not exist, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


if (len(sys.argv) < 1):
    raise Exception("Faltan parametros")

print(sys.argv)
scale = float(sys.argv[1])/100
print(scale)
tempCarpet = os.getenv('carpetaTemporal')
scaledFolder=os.getenv('scaledFolder')
savingFolder=f"{tempCarpet}\{scaledFolder}"
print(savingFolder)
ensure_directory_exists(savingFolder)

formats = []
for i in trimesh.available_formats():
    formats.append(f".{i}")

extensiones = tuple(formats)

NumFiles=0
with os.scandir(tempCarpet) as it:
    for entry in it:
        if entry.is_file() and entry.name.lower().endswith(extensiones):
            NumFiles=NumFiles+1

with os.scandir(tempCarpet) as it:
    for entry in it:
        if entry.is_file() and entry.name.lower().endswith(extensiones):
            print("Computando " + entry.path)
            (Sx, Sy, Sz, volumenEstimado, areaEstimada) = getInfoMesh(
                entry.path,savingFolder, scale,False)
            print(f'Tamaño(mm): {Sx} \t {Sy} \t {Sz} \t{volumenEstimado}')


