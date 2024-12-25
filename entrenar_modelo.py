import cv2
import os
import numpy as np
import time

# Ruta al dataset procesado
processedPath = '/Users/angelinaquidiazgonzalezrivas/Documents/Proyecto/Python/CV/Datasets/Procesado'  # Carpeta donde están "Hombre" y "Mujer"
clases = os.listdir(processedPath)  # Listará las carpetas "Hombre" y "Mujer"
print("Clases encontradas: ", clases)

labels = []
facesData = []
label = 0

# Leer las imágenes de cada clase
for clase in clases:
    clasePath = os.path.join(processedPath, clase)
    
    print(f"Leyendo imágenes de la clase: {clase}")
    time.sleep(2)
    flag = 0
    for fileName in os.listdir(clasePath):
        if flag <= 450:
            filePath = os.path.join(clasePath, fileName)
            print(f"Procesando archivo: {filePath}")
        
            # Cargar imagen en escala de grises
            imagen = cv2.imread(filePath, 0)
            if imagen is None:
                print(f"Error al cargar imagen: {filePath}")
                continue

            # Redimensionar la imagen a 150x150
            imagen = cv2.resize(imagen, (150, 150))
            facesData.append(imagen)
            labels.append(label)
            flag += 1

    label += 1

# Verificar equilibrio de las clases
print(f"Número de imágenes por clase: {np.bincount(labels)}")

cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV

# Entrenar el modelo EigenFaces
print("Entrenando modelo...")
reconocimientoGenero = cv2.face.EigenFaceRecognizer_create()
reconocimientoGenero.train(facesData, np.array(labels))

# Guardar el modelo entrenado
modeloPath = 'D:/Python/CV/modeloGenero.xml'
reconocimientoGenero.write(modeloPath)
print(f"Modelo almacenado en: {modeloPath}")
