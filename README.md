# Proyecto de riesgo crediticio

Este proyecto sirve para evaluar el riesgo crediticio de una persona usando un modelo de árbol de decisión.

También utiliza:

- FastAPI para realizar la predicción.
- Gradio para mostrar el chat.
- Ollama para entender los datos escritos por el usuario.
- MongoDB para guardar el historial de las evaluaciones.

## ¿Cómo funciona?

1. El usuario escribe sus datos en el chat.
2. Ollama identifica los datos proporcionados.
3. Si falta información, el chat la solicita.
4. El sistema calcula automáticamente las relaciones de deuda e importe del préstamo con los ingresos.
5. Cuando están completos los 15 datos, se envían a FastAPI.
6. El modelo devuelve una de estas categorías:

   0 = Riesgo bajo
   1 = Riesgo medio
   2 = Riesgo alto

7. En MongoDB se guarda el usuario, la predicción y las cinco características más importantes.

## Instalación

Primero hay que crear y activar el entorno virtual:

   python -m venv venv
   .\venv\Scripts\Activate.ps1

Después se instalan las librerías:

   pip install -r requirements.txt

También se necesita un archivo `.env` con la configuración:

   MONGODB_URI=tu_conexion_de_mongodb
   DATABASE_NAME=credit_risk_db
   COLLECTION_NAME=predictions

   FASTAPI_URL=http://127.0.0.1:8000

   OLLAMA_URL=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3.2

## Configuración de Ollama

Ollama debe estar instalado y ejecutándose en la computadora.

Para descargar el modelo:

   ollama pull llama3.2

Si PowerShell no reconoce el comando `ollama`, se puede usar:

   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.2

Para verificar los modelos instalados:

   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list

## Ejecutar el proyecto

Se necesitan dos terminales con el entorno virtual activado.

En la primera terminal se inicia FastAPI:

   uvicorn main:app --reload

La documentación de la API estará disponible en:

   http://127.0.0.1:8000/docs

En la segunda terminal se inicia la interfaz:

   python -m interfaces.chat_interface

La interfaz estará disponible en:

   http://127.0.0.1:7860

Desde ahí se puede comenzar a conversar con el asistente y realizar una evaluación crediticia.