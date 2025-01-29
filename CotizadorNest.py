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

from plates_manager import PlatesManager
from itemQuota import ItemQuota

load_dotenv()

def getInfoMesh(actualFile, try2Fix):
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
        area = mesh.area
        return Sx, Sy, Sz, volume, area
    except:
        return 0, 0, 0, 0, 0

def render3DModel(actualFile, carpet):
    fileName = os.path.basename(actualFile)  # Use actualFile instead of entry.path
    myMesh = pv.read(actualFile)
    
    p = pv.Plotter(off_screen=True, notebook=False)
    p.add_mesh(myMesh)
    
    # Set the background color to cyan
    p.background_color = 'cyan'
    
    photoName = os.path.splitext(fileName)[0]  # Use os.path.splitext to remove file extension
    p.screenshot(os.path.join(carpet, photoName + ".jpg"))
    return photoName


if (len(sys.argv) < 3):
    raise Exception("Faltan parametros")

df = pd.read_csv("resources\Materiales.csv")
print(sys.argv)
material = sys.argv[1]
infill = int(sys.argv[2])
infill = infill/100.0
try2Fix = (sys.argv[3] == 'True')
shipping_cost=float(sys.argv[4])/100

carpeta = os.getenv('carpetaTemporal')
infoMaterial = df[df["Material"] == material]
Densidad = infoMaterial['Densidad'].to_numpy()[0]
materialPrice = infoMaterial['PrecioGramo'].to_numpy()[0]
precioCarga = infoMaterial['PrecioCarga'].to_numpy()[0]
tipo=infoMaterial['Tipo'].to_string()[0]

print(
    f'Densidad:{Densidad},materialPrice:{materialPrice},precioCarga:{precioCarga}')

factorDeVelocidad = 180 #mm/s
AreaDeFilamentos = 1.75*1.75*3.14*300
print("Costo por gramo del material"+str(materialPrice))

FactorGanancia = 0.3
# Impuestos a considerar
FactorISR = 0.3
FactorIVA = 0.16
Impuestos = FactorISR+FactorIVA
FactorMul = (1+FactorGanancia)*(1+FactorISR+FactorIVA)

# Salario minimo
costoDia = 278.8

x = datetime.now()
year = x.year
month = x.month
day = x.day
totalSinIVA = 0
totalTiempo = 0
totalMaterial = 0
totalTiempo = 0
renderImage = True

formats = []
for i in trimesh.available_formats():
    formats.append(f".{i}")

extensiones = tuple(formats)

NumFiles=0
with os.scandir(carpeta) as it:
    for entry in it:
        if entry.is_file() and entry.name.lower().endswith(extensiones):
            NumFiles=NumFiles+1
print(f"Files: {NumFiles}")

extraPorLoad=precioCarga
if NumFiles<4:
    extraPorLoad=(precioCarga+shipping_cost)/NumFiles
else:
    extraPorLoad=(precioCarga+NumFiles*precioCarga/3+shipping_cost)/NumFiles
print(f"Costo por carga: {extraPorLoad}")

data = []

#Plate in cm
plate_x=22
plate_y=22
plate_z=22

if(tipo=='SLA'): 
    plate_x=21
    plate_y=12
    plate_z=24


Nester=PlatesManager(plate_x,plate_y)
Nester.create()
i=1
#Computations for FDM
with os.scandir(carpeta) as it:
    for entry in it:
        if entry.is_file() and entry.name.lower().endswith(extensiones):
            print("Computando " + entry.path)
            (Sx, Sy, Sz, volumenEstimado, areaEstimada) = getInfoMesh(
                entry.path, try2Fix)
            print(f'Tamaño(mm): {Sx} \t {Sy} \t {Sz} \t{volumenEstimado}')
            
            #arrange dimensiones using the max 
            Organized=int(np.sort(np.ceil([Sx,Sy,Sz]/10)))
            [Tx,Ty,Tz]=Organized
            Nester.find_and_place_rectangle(Tx,Ty,i)
            
            materialParedes =areaEstimada*1.2/1000
            area_por_capa = areaEstimada/(2*3*(10*10))

            #gr 
            materialNecesario = (volumenEstimado/1000 * 1.61*infill+materialParedes)*Densidad
            if(tipo=='SLA'): #ml
                materialNecesario = (Sx/10*Sy/10*Sz/10)*(1.61)*infill+materialParedes
            totalMaterial += materialNecesario
            costoMaterial = materialNecesario*materialPrice

            #Calculando el tiempo en segundos luego lo pasaremos a minutos  
            h=0.16
            tiempoCurado=0.1 #Layer change
            #FDM 0.4mm nozzle 200 mm/s
            tiempoMovimiento=Tx/10*Ty/10*1/(0.4*200)
            #Calentamiento más nivelado
            tiempoDeFondo=10*60
            if(tipo=='SLA'): 
                h=0.05
                tiempoCurado=3
                tiempoMovimiento=2
                tiempoDeFondo=240
                
            layers=(Tz/10+3.5)/h
            
            timePerLayer=layers*(tiempoCurado+tiempoMovimiento)
            tiempoEstimado = (timePerLayer+240)/60 #min
            tiempoDeLimpieza = max(tiempoEstimado/14, 5)  
            
            #Ajuste debido a multiples partes por la altura
            tiempoDeFondo=tiempoDeFondo*np.ceil(Tz/plate_z)  
            tiempoDeLimpieza=tiempoDeFondo*np.ceil(Tz/plate_z)  
            
            costoTiempo = (tiempoDeLimpieza) * (costoDia)/(8*60)
            costo = costoTiempo
                        
            print(f'Tiempo: {tiempoEstimado} \t Limpieza: {tiempoDeLimpieza} \t {Tx} \t {Ty} \t{Tz} \t{volumenEstimado}')
                
            CostoAntesDeIVA = costo
            CostoConIVA = 0
            totalSinIVA += 0
            totalTiempo += tiempoEstimado
            fileName = os.path.basename(entry.path)
            infod = ItemQuota(fileName, Sx, Sy, Sz, areaEstimada, volumenEstimado, material, infill,
                            materialNecesario, tiempoEstimado, CostoAntesDeIVA, CostoConIVA)
            infod.id=i
            i=i+1
            if renderImage:
                photoName = render3DModel(entry.path, carpeta)
                infod.setImage(photoName)
            data.append(infod)
            
            if renderImage:
                photoName = render3DModel(entry.path, carpeta)
                infod.setImage(photoName)
            data.append(infod)

                
Nester.print_plates()
dfSLA = pd.DataFrame([[
    x.Name, x.Sx, x.Sy, x.Sz, x.Volumen,
    x.TipoDeMaterial, x.Relleno, x.MaterialRequerido, x.Tiempo, x.PrecioSinIVA,x.PrecioConIVA ,x.id] for x in data],
    columns=['Nombre', 'Sx', 'Sy', 'Sz', 'Volumen',
            'Material', 'Relleno', 'MaterialEstimado', 'TiempoEstimado', 'CostoAntesDeIVA','CostoConIVA' ,'id'])


#Full analysis of the plates
UnionDePlatos=np.array(Nester.plates).flatten()
RemocionDeZeros = UnionDePlatos[UnionDePlatos != 0]
Identificadores, Cantidad = np.unique(RemocionDeZeros, return_counts=True)
Resumen = dict(zip(Identificadores, Cantidad))

for plate in Nester.plates:
    values = np.unique(plate)
    items = len(values)
    print(f'Plato:  {values}')
    if items == 0:
        print("Brincando primer plato")
        continue
    timeModel = 0
    PlateTime = 0

    for id in values:
        
        if id == 0:
            continue

        # Filter the DataFrame
        result = dfSLA[dfSLA["id"] == id]

        if not result.empty:
            tiempo_estimado = result['TiempoEstimado'].iloc[0]
            #print(result)

            if timeModel < tiempo_estimado:
                PlateTime = tiempo_estimado
        else:
            print(f"No matching record found in dfSLA for id={id}")

    # Cost calculations
    fepPrice = 890 * 1.16 / 10
    alcohol = 30
    
    costo = (timeModel * costoDia / (8 * 60) + fepPrice + alcohol) / items+shipping_cost/NumFiles
    print(f'Cost added per Part of {costo}')

    # Update `data` objects
    for id in values:
        print(f'Time Analysis of {id}')
        matching_object = next((obj for obj in data if obj.id == id), None)

        if matching_object:
            print(f'Original Price of {matching_object.PrecioSinIVA}')
            matching_object.PrecioSinIVA = (matching_object.PrecioSinIVA + costo) * (1+FactorGanancia) * (1 + FactorISR)
            print(f'Updated Price of {matching_object.PrecioSinIVA}')
            matching_object.PrecioConIVA = matching_object.PrecioSinIVA * (1 + FactorIVA)
        else:
            print(f"No matching object found in data for id={id}")
            
                    
df = pd.DataFrame([[
    x.Name, x.Sx, x.Sy, x.Sz, x.Volumen,
    x.TipoDeMaterial, x.Relleno, x.MaterialRequerido, x.Tiempo, x.PrecioSinIVA, x.PrecioConIVA] for x in data],
    columns=['Nombre', 'Sx', 'Sy', 'Sz', 'Volumen',
             'Material', 'Relleno', 'MaterialEstimado', 'TiempoEstimado', 'CostoAntesDeIVA', 'CostoConIVA'])

df.to_csv(carpeta+"\\info.csv", index=False)

easy2read = df[['Nombre', 'Material', 'Relleno',
                'CostoAntesDeIVA', 'CostoConIVA']]
easy2read.to_csv(carpeta + "\\"+'easy.csv', index=False)


# Create empty list to store images
images = []

# Loop through images in the folder and append to the list
for filename in os.listdir(carpeta):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        filepath = os.path.join(carpeta, filename)
        images.append(Image.open(filepath))

# Specify the aurum number (golden ratio) to determine the layout
aurum_number = 1.618/2 # Adjust this value based on your preference

# Compute the number of columns based on the aurum number
num_images = len(images)
num_columns = max(int(num_images / aurum_number), 1)
num_rows = -(-num_images // num_columns)  # Ceiling division to get the number of rows

# Determine the maximum width and height of the final image
max_width = max(image.size[0] for image in images)
max_height = max(image.size[1] for image in images)
total_width = max_width * num_columns
total_height = max_height * num_rows

# Scale down dimensions if they exceed the 65,500 limit
max_dimension = 65500
scale_factor = min(1, max_dimension / max(total_width, total_height))
scaled_width = int(total_width * scale_factor)
scaled_height = int(total_height * scale_factor)

# Create a new image with the determined size
new_image = Image.new("RGB", (scaled_width, scaled_height))

# Adjust individual image dimensions according to the scale factor
scaled_max_width = int(max_width * scale_factor)
scaled_max_height = int(max_height * scale_factor)

# Paste images onto the new image in a mosaic pattern
x_offset, y_offset = 0, 0
for idx, image in enumerate(images):
    resized_image = image.resize((scaled_max_width, scaled_max_height))
    new_image.paste(resized_image, (x_offset, y_offset))
    y_offset += scaled_max_height
    if (idx + 1) % num_rows == 0:  # Move to the next column after filling the current column
        y_offset = 0
        x_offset += scaled_max_width

# Optionally resize the final mosaic further (if needed)
final_image = new_image.resize(
    (int(new_image.size[0] / 2), int(new_image.size[1] / 2)))

# Save the final image
output_path = os.path.join(carpeta, "joined_image.jpg")
final_image.save(output_path)