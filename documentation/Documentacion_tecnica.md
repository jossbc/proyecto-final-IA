# Documentación técnica

## Descripción

Sistema local de evaluación de riesgo crediticio compuesto por:

- Un modelo `DecisionTreeClassifier`.
- Una API REST desarrollada con FastAPI.
- Un agente conversacional con Gradio y Ollama.
- MongoDB para registrar el historial de predicciones.

La salida del modelo es una categoría:

| Valor | Categoría |
|---:|---|
| 0 | Riesgo bajo |
| 1 | Riesgo medio |
| 2 | Riesgo alto |

## Arquitectura

Usuario
  → Gradio
  → Servicio de conversación
  → Ollama (extracción de datos)
  → FastAPI /predict
  → Árbol de decisión
  → MongoDB
  → Resultado en Gradio

Ollama se utiliza únicamente para extraer datos del lenguaje natural. La
clasificación de riesgo siempre es realizada por el árbol de decisión.

## Estructura del proyecto

```text
artifacts/       Modelo entrenado y metadatos
controllers/     Lógica de predicción y persistencia
data/            Dataset utilizado
documentation/   Manual y documentación
interfaces/      Interfaz conversacional de Gradio
models/          Esquemas y validaciones de Pydantic
notebook/        Preparación, entrenamiento y evaluación
routes/          Endpoints de FastAPI
services/        Comunicación con Ollama y FastAPI
utils/           Conexión con MongoDB
main.py          Punto de entrada de FastAPI
```

### Archivos principales

- `notebook/risk_model.ipynb`: limpieza, análisis, entrenamiento, evaluación y
  exportación del modelo.
- `artifacts/credit_risk_model.pkl`: contiene el modelo, los nombres de las 15
  características y las cinco características más importantes.
- `models/credit_application.py`: define los esquemas de entrada, extracción y
  respuesta.
- `routes/prediction_routes.py`: define `POST /predict`.
- `controllers/prediction_controller.py`: ejecuta la predicción y registra el
  resultado.
- `services/ollama_service.py`: extrae datos desde mensajes mediante Ollama.
- `services/conversation_service.py`: administra el estado de la conversación.
- `services/api_client.py`: consume la API desde el agente.
- `interfaces/chat_interface.py`: construye la interfaz de Gradio.
- `utils/mongo.py`: administra la conexión con MongoDB.

## Modelo predictivo

Se utilizó un árbol de decisión.
El dataset se dividió en 80 % para entrenamiento y 20 % para prueba, utilizando
estratificación. El modelo se evaluó mediante accuracy, precision, recall,
F1-score y matriz de confusión.

No se aplicó escalado porque los árboles de decisión no dependen de distancias
entre observaciones. Los valores atípicos fueron analizados mediante IQR y se
conservaron al no encontrarse valores inválidos.

### Características

El modelo utiliza 15 características:

```text
age
occupation_status
years_employed
annual_income
credit_score
credit_history_years
savings_assets
current_debt
defaults_on_file
delinquencies_last_2yrs
derogatory_marks
loan_intent
loan_amount
debt_to_income_ratio
loan_to_income_ratio
```

Los ratios se calculan antes de consumir la API:

```text
debt_to_income_ratio = current_debt / annual_income
loan_to_income_ratio = loan_amount / annual_income
```

Las cinco características principales guardadas en MongoDB son:

```text
credit_score
loan_to_income_ratio
debt_to_income_ratio
delinquencies_last_2yrs
derogatory_marks
```

## API

### `POST /predict`

Recibe un objeto `CreditApplication` con el identificador y las 15
características.

Ejemplo de respuesta:

```json
{
  "risk_level": 0
}
```

La API no devuelve score, probabilidades ni porcentajes.

Pydantic valida tipos y rangos antes de ejecutar el modelo. Entre las
validaciones principales se encuentran:

- `credit_score`: entre 300 y 850.
- `occupation_status`: entre 0 y 2.
- `loan_intent`: entre 0 y 5.
- `defaults_on_file`: 0 o 1.
- Ingresos y préstamo: mayores que cero.
- Deudas, ahorros, ratios y años: valores no negativos.

## Agente conversacional

El agente mantiene un estado con:

```python
{
    'data': {},
    'pending_field': None,
    'completed': False
}
```

Ollama recibe el mensaje, los datos previamente recopilados y un esquema JSON.
El resultado se valida con `ExtractedApplicantData`.

Cuando falta información, el agente solicita el primer campo pendiente. Cuando
los datos están completos, construye el payload, consume `/predict` y aplica
estas reglas:

| Categoría | Decisión |
|---:|---|
| 0 | Aprobar con condiciones estándar |
| 1 | Solicitar documentación adicional |
| 2 | Rechazar y recomendar educación financiera |

## Persistencia

Cada evaluación se guarda en MongoDB con la siguiente estructura:

```json
{
  "user_id": "cliente-001",
  "features": {
    "credit_score": 720,
    "loan_to_income_ratio": 0.2,
    "debt_to_income_ratio": 0.083,
    "delinquencies_last_2yrs": 0,
    "derogatory_marks": 0
  },
  "prediction": 0
}
```

El modelo recibe las 15 características, pero el registro conserva únicamente
las cinco más importantes.

## Variables de entorno

```env
MONGODB_URI=mongodb://...
DATABASE_NAME=credit_risk_db
COLLECTION_NAME=predictions

FASTAPI_URL=http://127.0.0.1:8000

OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

## Ejecución

API:

```powershell
uvicorn main:app --reload
```

Interfaz:

```powershell
python -m interfaces.chat_interface
```

Servicios:

```text
FastAPI: http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs
Gradio:  http://127.0.0.1:7860
Ollama:  http://127.0.0.1:11434
```

## Manejo de errores

- FastAPI devuelve errores de validación cuando el payload es inválido.
- El agente controla fallos de conexión con FastAPI y Ollama.
- El controlador devuelve `503` cuando no puede registrar la evaluación.
- La categoría se valida en la API y en el cliente; solo se aceptan 0, 1 y 2.

 