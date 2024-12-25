import cv2
import os
from nltk.chat.util import Chat, reflections
import sys  # Para salir del programa completamente

# Ruta al modelo entrenado
modeloPath = '/Users/angelinaquidiazgonzalezrivas/Documents/Proyecto/ReconocimientoFa/modeloGenero.xml'
processedPath = '/Users/angelinaquidiazgonzalezrivas/Documents/Proyecto/Python/CV/Datasets/Procesado'
clases = os.listdir(processedPath)  # Lee las clases ("Hombre", "Mujer")

# Cargar el modelo entrenado
reconocimientoGenero = cv2.face.EigenFaceRecognizer_create()
reconocimientoGenero.read(modeloPath)

# Inicializar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
clasificadorRostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuración del chatbot
pares_hombre = [
    [r"hola|hey|buenas", ["Hola, caballero. ¿Cómo estás?"]],
    [r"como estas?", ["Estoy bien, ¿y tú?"]],
    [r"que haces?", ["Converso contigo."]],
    [r"finalizar", ["Adiós, fue un gusto hablar contigo."]]
]

pares_mujer = [
    [r"hola|hey|buenas", ["Hola, señora/señorita. ¿Cómo te va?"]],
    [r"como estas?", ["Estoy bien, ¿y tú?"]],
    [r"que haces?", ["Estoy aquí para conversar contigo."]],
    [r"finalizar", ["Adiós, espero volver a hablar contigo pronto."]]
]

# Función para iniciar el chatbot
def iniciar_chat(pares):
    chat = Chat(pares, reflections)
    print("Escribe algo para comenzar. Escribe 'finalizar' para salir.")
    
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "finalizar":
            print("Chatbot: Adiós.")
            sys.exit()  # Termina el programa completamente
        
        response = chat.respond(user_input)
        if response:
            print(f"Chatbot: {response}")
        else:
            print("Chatbot: Interesante, cuéntame más.")  # Respuesta genérica para entradas no reconocidas

# Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    # Detectar rostros en la imagen
    caras = clasificadorRostro.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in caras:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Predecir género
        resultado = reconocimientoGenero.predict(rostro)
        genero = clases[resultado[0]]
        confianza = resultado[1]

        # Mostrar resultado en pantalla
        if confianza < 8000:  # Ajusta este umbral según el modelo
            cv2.putText(frame, f"{genero} ({confianza:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Activar chatbot según el género
            if genero == "Hombre":
                print("Detecté a un hombre. Activando chatbot para hombre.")
                iniciar_chat(pares_hombre)
            elif genero == "Mujer":
                print("Detecté a una mujer. Activando chatbot para mujer.")
                iniciar_chat(pares_mujer)
        else:
            cv2.putText(frame, "Desconocido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Reconocimiento de Género', frame)

    if cv2.waitKey(1) == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
