import os
import scipy.io
import shutil
import numpy as np

# Ruta al dataset y archivo .mat
datasetPath = '/Users/angelinaquidiazgonzalezrivas/Documents/Proyecto/Python/CV/Datasets/IMDB-WIKI/wiki_crop/'
labelsFile = '/Users/angelinaquidiazgonzalezrivas/Documents/Proyecto/Python/CV/Datasets/IMDB-WIKI/wiki_crop/wiki.mat'

# Verificar que el archivo .mat exista
if not os.path.exists(labelsFile):
    raise FileNotFoundError(f"El archivo wiki.mat no se encuentra en: {labelsFile}")

# Carpetas de destino
processedPath = 'D:/Python/CV/Datasets/Procesado'
hombrePath = os.path.join(processedPath, 'Hombre')
mujerPath = os.path.join(processedPath, 'Mujer')
os.makedirs(hombrePath, exist_ok=True)
os.makedirs(mujerPath, exist_ok=True)

# Cargar el archivo .mat
mat = scipy.io.loadmat(labelsFile)

# Ajustar las claves
gender = mat['wiki']['gender'][0][0][0]  # Género (1 = hombre, 0 = mujer)
imagePaths = mat['wiki']['full_path'][0][0][0]  # Rutas relativas

# Depurar rutas
print("Primeras 10 rutas relativas del archivo .mat:")
print([str(path[0]) for path in imagePaths[:10]])

# Contadores
male_count = 0
female_count = 0
MAX_IMAGES_PER_GENDER = 600

# Procesar imágenes
for i, imgPath in enumerate(imagePaths):
    try:
        if male_count >= MAX_IMAGES_PER_GENDER and female_count >= MAX_IMAGES_PER_GENDER:
            break

        # Convertir la ruta relativa a una cadena estándar de Python
        relativePath = str(imgPath[0])  # Limpiar el formato de NumPy
        imgFullPath = os.path.join(datasetPath, relativePath.replace('\\', '/'))  # Normalizar separadores

        if not os.path.exists(imgFullPath):
            print(f"Imagen encontrada: {imgFullPath}")
            continue

        if i >= len(gender) or np.isnan(gender[i]):
            print(f"Género inválido en índice {i}")
            continue

        genderLabel = int(gender[i])  # Género
        if genderLabel == 1 and male_count < MAX_IMAGES_PER_GENDER:
            shutil.copy(imgFullPath, os.path.join(hombrePath, os.path.basename(imgFullPath)))
            male_count += 1
        elif genderLabel == 0 and female_count < MAX_IMAGES_PER_GENDER:
            shutil.copy(imgFullPath, os.path.join(mujerPath, os.path.basename(imgFullPath)))
            female_count += 1

    except Exception as e:
        print(f"Error procesando la imagen {imgPath}: {e}")

print("Clasificación completada.")
print(f"Imágenes procesadas: {male_count} hombres, {female_count} mujeres.")
print("Imágenes guardadas en:", processedPath)
