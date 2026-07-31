import json
import os

import httpx
from dotenv import load_dotenv

from models.credit_application import ExtractedApplicantData


load_dotenv()

OLLAMA_URL = os.getenv('OLLAMA_URL')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')


class OllamaServiceError(RuntimeError):
    pass


ZERO_PHRASES = {
    'defaults_on_file': (
        'nunca he incumplido',
        'no he incumplido',
        'sin incumplimientos'
    ),
    'delinquencies_last_2yrs': (
        'no he tenido atrasos',
        'no tengo atrasos',
        'sin atrasos',
        'ningún atraso',
        'ningun atraso'
    ),
    'derogatory_marks': (
        'no tengo marcas negativas',
        'sin marcas negativas',
        'ninguna marca negativa',
        'no tengo marcas derogatorias'
    )
}


def extract_explicit_zeros(message: str) -> dict:
    text = message.lower()

    return {
        field: 0
        for field, phrases in ZERO_PHRASES.items()
        if any(phrase in text for phrase in phrases)
    }


def extract_pending_value(
    message: str,
    pending_field: str | None
) -> dict:
    if not pending_field:
        return {}

    text = message.strip().lower()

    zero_count_fields = {
        'delinquencies_last_2yrs',
        'derogatory_marks'
    }

    if (
        pending_field in zero_count_fields
        and text in {'no', 'ninguno', 'ninguna', 'cero', '0'}
    ):
        return {pending_field: 0}

    numeric_fields = {
        'age': int,
        'years_employed': float,
        'annual_income': float,
        'credit_score': int,
        'credit_history_years': float,
        'savings_assets': float,
        'current_debt': float,
        'delinquencies_last_2yrs': int,
        'derogatory_marks': int,
        'loan_amount': float
    }

    if pending_field in numeric_fields:
        number = text.replace(',', '.').split()[0]

        try:
            value_type = numeric_fields[pending_field]
            return {
                pending_field: value_type(float(number))
            }
        except ValueError:
            return {}

    if pending_field == 'defaults_on_file':
        if text in {'no', 'nunca', 'ninguno', 'ninguna', '0'}:
            return {'defaults_on_file': 0}

        if text in {'sí', 'si', '1'}:
            return {'defaults_on_file': 1}

    if pending_field == 'user_id' and len(text.split()) <= 5:
        return {'user_id': message.strip()}

    return {}


def extract_applicant_data(
    user_message: str,
    collected_data: dict,
    pending_field: str | None = None
) -> dict:
    response_schema = ExtractedApplicantData.model_json_schema()
    response_schema['required'] = list(
        response_schema['properties']
    )
    response_schema['additionalProperties'] = False

    prompt = f'''
Extrae todos los datos crediticios mencionados por el usuario.

Reglas:
- No inventes datos.
- Si un dato no aparece, devuélvelo como null.
- Lee el mensaje completo y relaciona cada cifra con la frase que la acompaña.
- No confundas edad, años trabajando e historial crediticio.
- No confundas ingreso, ahorros, deuda y cantidad del préstamo.
- "No", "nunca" o "ninguno" representan 0 cuando se habla de
  incumplimientos, atrasos o marcas negativas.
- Si existe un campo solicitado anteriormente, interpreta primero el mensaje
  como respuesta a ese campo.
- Devuelve únicamente JSON.

Campos:
- user_id: nombre o identificador.
- age: edad de la persona.
- occupation_status: Employed, Student o Self-Employed.
- years_employed: años que lleva trabajando.
- annual_income: ingreso anual.
- credit_score: puntaje crediticio.
- credit_history_years: años de historial crediticio.
- savings_assets: ahorros y activos.
- current_debt: deuda actual.
- defaults_on_file: 1 si ha incumplido; 0 si nunca ha incumplido.
- delinquencies_last_2yrs: atrasos en los últimos dos años.
- derogatory_marks: marcas negativas.
- loan_intent: Business, Home Improvement, Debt Consolidation, Education,
  Personal o Medical.
- loan_amount: cantidad solicitada.

Datos anteriores:
{json.dumps(collected_data, ensure_ascii=False)}

Campo solicitado anteriormente:
{pending_field}

Mensaje:
{user_message}
'''

    payload = {
        'model': OLLAMA_MODEL,
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'stream': False,
        'format': response_schema,
        'options': {
            'temperature': 0
        }
    }

    try:
        response = httpx.post(
            f'{OLLAMA_URL.rstrip("/")}/api/chat',
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        content = response.json()['message']['content']
        extracted_json = json.loads(content)

        validated = ExtractedApplicantData(**extracted_json)

        extracted_data = validated.model_dump(exclude_none=True)

        explicit_zeros = extract_explicit_zeros(user_message)
        for field, value in explicit_zeros.items():
            extracted_data.setdefault(field, value)

        if pending_field not in extracted_data:
            extracted_data.update(
                extract_pending_value(
                    user_message,
                    pending_field
                )
            )

        return extracted_data

    except (
        httpx.HTTPError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError
    ) as error:
        raise OllamaServiceError(
            'No fue posible extraer los datos mediante Ollama.'
        ) from error
