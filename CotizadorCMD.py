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

class item2Quota:
    def __init__(self, Name, Sx, Sy, Sz, Area, Volumen, TipoDeMaterial, Relleno, MaterialRequerido, Tiempo, PrecioSinIVA, PrecioConIVA):
        self.Name = Name
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz
        self.Area = Area
        self.Volumen = Volumen
        self.TipoDeMaterial = TipoDeMaterial
        self.Relleno = Relleno
        self.MaterialRequerido = MaterialRequerido
        self.Tiempo = Tiempo
        self.PrecioSinIVA = PrecioSinIVA
        self.PrecioConIVA = PrecioConIVA
        self.Imagen = ""

    def setImage(self, Imagen):
        self.Imagen = Imagen

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

print(
    f'Densidad:{Densidad},materialPrice:{materialPrice},precioCarga:{precioCarga}')

factorDeVelocidad = 60
AreaDeFilamentos = 1.75*1.75*3.14*300
print("Costo por gramo del material"+str(materialPrice))

FactorGanancia = 0.45
# Impuestos a considerar
FactorISR = 0.3
FactorIVA = 0.16
Impuestos = FactorISR+FactorIVA
FactorMul = (1+FactorGanancia)*(1+FactorISR+FactorIVA)
# Costo del día operable por equipo
costoDia = 256

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
print(f"extra per load: {extraPorLoad}")
data = []
with os.scandir(carpeta) as it:
    for entry in it:
        if entry.is_file() and entry.name.lower().endswith(extensiones):
            print("Computando " + entry.path)
            (Sx, Sy, Sz, volumenEstimado, areaEstimada) = getInfoMesh(
                entry.path, try2Fix)
            print(f'Tamaño(mm): {Sx} \t {Sy} \t {Sz} \t{volumenEstimado}')
            Paredes = 3
            area_por_capa = areaEstimada/(2*3*(10*10))
            tiempoVolumen = volumenEstimado/1000*infill*factorDeVelocidad/60*2
            tiempoParedes = area_por_capa*factorDeVelocidad/60*4
            tiempoEstimado = tiempoVolumen+tiempoParedes
            materialNecesario = volumenEstimado/1000 * \
                infill*0.4+area_por_capa*0.4*Paredes*0.6
            totalMaterial += materialNecesario
            costoMaterial = materialNecesario*materialPrice
            costoTiempo = (tiempoParedes+tiempoVolumen) * \
                (costoDia*FactorGanancia)/(10*60)
            costo = costoMaterial+costoTiempo+extraPorLoad
            CostoAntesDeIVA = costo*(1+FactorGanancia+FactorISR)
            CostoConIVA = CostoAntesDeIVA*(1+FactorIVA)
            totalSinIVA += CostoAntesDeIVA
            totalTiempo += tiempoEstimado
            fileName = os.path.basename(entry.path)
            infod = item2Quota(fileName, Sx, Sy, Sz, areaEstimada, volumenEstimado, material, infill,
                               materialNecesario, tiempoEstimado, CostoAntesDeIVA, CostoConIVA)
            if renderImage:
                photoName = render3DModel(entry.path, carpeta)
                infod.setImage(photoName)
            data.append(infod)

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
# Loop through images in folder and append to list
numImages = 0
images = []
for filename in os.listdir(carpeta):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        filepath = os.path.join(carpeta, filename)
        images.append(Image.open(filepath))
        numImages += 1

# Specify the aurum number (replace this with the actual value), that can generate the a longer image
aurum_number = 1.618/2  # You can adjust this value based on your preference

# Compute the number of rows based on the aurum number
num_rows = max(int(numImages / aurum_number), 1)

# Determine size of final image
max_width = max(image.size[0] for image in images)
total_height = sum(image.size[1] for image in images) // num_rows

# Create a new image with determined size
new_image = Image.new('RGB', (max_width, total_height))

# Paste images onto new image in a mosaic pattern
x_offset, y_offset = 0, 0
for image in images:
    new_image.paste(image, (x_offset, y_offset))
    y_offset += image.size[1]
    if y_offset >= total_height:
        y_offset = 0
        x_offset += max_width

# Save the final image
new_image.save(os.path.join(carpeta, 'joined_image.jpg'))