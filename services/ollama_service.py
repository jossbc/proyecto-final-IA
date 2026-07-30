import json
import os
import re
import unicodedata

import httpx
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from models.credit_application import ExtractedApplicantData


load_dotenv()

OLLAMA_URL = os.getenv('OLLAMA_URL')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')


class OllamaServiceError(RuntimeError):
    pass


FLOAT_FIELDS = {
    'years_employed',
    'annual_income',
    'credit_history_years',
    'savings_assets',
    'current_debt',
    'loan_amount'
}

INTEGER_FIELDS = {
    'age',
    'credit_score',
    'delinquencies_last_2yrs',
    'derogatory_marks'
}

FIELD_GUIDE = {
    'user_id': 'Nombre o identificador del usuario.',
    'age': 'Edad de la persona, no años de trabajo ni historial.',
    'occupation_status': (
        'Employed si es empleado, Student si es estudiante o '
        'Self-Employed si trabaja de forma independiente.'
    ),
    'years_employed': 'Cantidad de años que lleva trabajando.',
    'annual_income': 'Ingreso anual de la persona.',
    'credit_score': 'Puntaje crediticio entre 300 y 850.',
    'credit_history_years': 'Años de historial crediticio.',
    'savings_assets': 'Cantidad indicada como ahorros o activos.',
    'current_debt': 'Cantidad indicada como deuda actual.',
    'defaults_on_file': (
        '1 si ha incumplido un crédito; 0 si dice nunca o no.'
    ),
    'delinquencies_last_2yrs': (
        'Número de atrasos en los últimos dos años; ninguno significa 0.'
    ),
    'derogatory_marks': (
        'Número de marcas negativas; ninguna significa 0.'
    ),
    'loan_intent': (
        'Motivo normalizado a Business, Home Improvement, '
        'Debt Consolidation, Education, Personal o Medical.'
    ),
    'loan_amount': 'Cantidad de dinero que desea solicitar.'
}


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize('NFD', value.lower())
    return ''.join(
        character
        for character in normalized
        if unicodedata.category(character) != 'Mn'
    )


def _fallback_pending_value(
    message: str,
    pending_field: str | None
) -> dict:
    if not pending_field:
        return {}

    normalized = _normalized_text(message).strip()
    number_match = re.search(r'-?\d+(?:[.,]\d+)?', normalized)

    if pending_field in FLOAT_FIELDS and number_match:
        value = float(number_match.group().replace(',', '.'))
        return {pending_field: value}

    if pending_field in INTEGER_FIELDS and number_match:
        value = int(float(number_match.group().replace(',', '.')))
        return {pending_field: value}

    if pending_field == 'defaults_on_file':
        if any(word in normalized for word in ('no', 'nunca', 'ninguno')):
            return {pending_field: 0}
        if any(word in normalized for word in ('si', 'incumpli')):
            return {pending_field: 1}

    if pending_field in {
        'delinquencies_last_2yrs',
        'derogatory_marks'
    } and any(
        word in normalized
        for word in ('ninguno', 'ninguna', 'cero')
    ):
        return {pending_field: 0}

    if pending_field == 'occupation_status':
        if 'estudiante' in normalized or 'student' in normalized:
            return {pending_field: 'Student'}
        if 'independiente' in normalized or 'self-employed' in normalized:
            return {pending_field: 'Self-Employed'}
        if 'empleado' in normalized or 'employed' in normalized:
            return {pending_field: 'Employed'}

    if pending_field == 'loan_intent':
        intent_terms = {
            'negocio': 'Business',
            'business': 'Business',
            'vivienda': 'Home Improvement',
            'deuda': 'Debt Consolidation',
            'educacion': 'Education',
            'personal': 'Personal',
            'medico': 'Medical',
            'medical': 'Medical'
        }
        for term, intent in intent_terms.items():
            if term in normalized:
                return {pending_field: intent}

    if pending_field == 'user_id' and message.strip():
        return {pending_field: message.strip()}

    return {}


def _fallback_explicit_negatives(
    message: str
) -> dict:
    normalized = _normalized_text(message)
    extracted = {}

    default_phrases = (
        'nunca he incumplido',
        'nunca incumpli',
        'no he incumplido',
        'sin incumplimientos'
    )
    if any(phrase in normalized for phrase in default_phrases):
        extracted['defaults_on_file'] = 0

    delinquency_phrases = (
        'no he tenido atrasos',
        'no tengo atrasos',
        'sin atrasos',
        'ningun atraso',
        'ninguna mora'
    )
    if any(phrase in normalized for phrase in delinquency_phrases):
        extracted['delinquencies_last_2yrs'] = 0

    mark_phrases = (
        'no tengo marcas negativas',
        'sin marcas negativas',
        'ninguna marca negativa',
        'no tengo marcas derogatorias',
        'sin marcas derogatorias'
    )
    if any(phrase in normalized for phrase in mark_phrases):
        extracted['derogatory_marks'] = 0

    return extracted


def _message_dicts(messages: list) -> list[dict[str, str]]:
    return [
        {
            'role': (
                'system'
                if message.type == 'system'
                else 'user'
            ),
            'content': str(message.content)
        }
        for message in messages
    ]


def _chat(
    messages: list[dict[str, str]],
    response_format: dict | str | None = None
) -> str:
    if not OLLAMA_URL or not OLLAMA_MODEL:
        raise OllamaServiceError(
            'Falta configurar OLLAMA_URL u OLLAMA_MODEL en .env.'
        )

    payload = {
        'model': OLLAMA_MODEL,
        'messages': messages,
        'stream': False,
        'options': {
            'temperature': 0
        }
    }

    if response_format is not None:
        payload['format'] = response_format

    try:
        response = httpx.post(
            f'{OLLAMA_URL.rstrip('/')}/api/chat',
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return str(response.json()['message']['content']).strip()
    except (httpx.HTTPError, KeyError, TypeError) as error:
        raise OllamaServiceError(
            'No fue posible comunicarse con Ollama. '
            'Verifica que esté iniciado y que el modelo configurado exista.'
        ) from error


def _extract_with_schema(
    user_message: str,
    collected_data: dict,
    pending_field: str | None,
    target_fields: list[str]
) -> dict:
    complete_schema = ExtractedApplicantData.model_json_schema()
    properties = complete_schema['properties']

    target_schema = {
        'type': 'object',
        'properties': {
            field: properties[field]
            for field in target_fields
        },
        'required': target_fields,
        'additionalProperties': False
    }

    target_guide = {
        field: FIELD_GUIDE[field]
        for field in target_fields
    }

    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            '''
Eres un extractor de datos para una evaluación crediticia.
Lee cuidadosamente todo el mensaje antes de responder.
Extrae únicamente información expresada por el usuario.
No inventes ni estimes valores.

Reglas:
- Asocia cada cifra con la frase que la acompaña.
- 'Nunca', 'no' o 'ninguno' puede representar 0.
- No confundas edad, años de trabajo y años de historial crediticio.
- No confundas ingreso, ahorros, deuda y cantidad del préstamo.
- Devuelve solamente JSON válido ajustado al esquema.

Guía de campos:
{field_guide}

Esquema JSON:
{json_schema}
            '''.strip()
        ),
        (
            'human',
            '''
Datos recopilados previamente:
{collected_data}

Dato solicitado en la pregunta anterior:
{pending_field}

Mensaje completo del usuario:
{user_message}

Busca los campos solicitados en todo el mensaje. Si alguno no fue mencionado,
devuélvelo como null.
            '''.strip()
        )
    ])

    messages = prompt.format_messages(
        field_guide=json.dumps(
            target_guide,
            ensure_ascii=False
        ),
        json_schema=json.dumps(
            target_schema,
            ensure_ascii=False
        ),
        collected_data=json.dumps(
            collected_data,
            ensure_ascii=False
        ),
        pending_field=pending_field or 'ninguno',
        user_message=user_message
    )

    content = _chat(
        _message_dicts(messages),
        response_format=target_schema
    )

    extracted_json = json.loads(content)
    validated = ExtractedApplicantData(**extracted_json)
    return validated.model_dump(exclude_none=True)


def extract_applicant_data(
    user_message: str,
    collected_data: dict,
    pending_field: str | None = None
) -> dict:
    all_fields = list(ExtractedApplicantData.model_fields)

    extracted_data = _extract_with_schema(
        user_message=user_message,
        collected_data=collected_data,
        pending_field=pending_field,
        target_fields=all_fields
    )

    missing_fields = [
        field
        for field in all_fields
        if field not in collected_data
        and field not in extracted_data
    ]

    is_detailed_message = (
        pending_field is None
        and len(user_message.split()) >= 25
        and len(extracted_data) >= 3
    )

    if is_detailed_message and missing_fields:
        retry_data = _extract_with_schema(
            user_message=user_message,
            collected_data={
                **collected_data,
                **extracted_data
            },
            pending_field=None,
            target_fields=missing_fields
        )
        extracted_data.update(retry_data)

    negative_values = _fallback_explicit_negatives(user_message)
    for field, value in negative_values.items():
        if field not in extracted_data:
            extracted_data[field] = value

    if pending_field not in extracted_data:
        fallback_data = _fallback_pending_value(
            user_message,
            pending_field
        )
        if fallback_data:
            validated_fallback = ExtractedApplicantData(
                **fallback_data
            )
            extracted_data.update(
                validated_fallback.model_dump(exclude_none=True)
            )

    return extracted_data
