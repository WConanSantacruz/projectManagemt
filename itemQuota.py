class ItemQuota:
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
        self.id=0

    def setImage(self, Imagen):
        self.Imagen = Imagen
    
    def returnSize(self):
        return self.Sx, self.Sz, self.Identificator