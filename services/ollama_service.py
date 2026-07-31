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


def extract_applicant_data(
    user_message: str,
    collected_data: dict,
    pending_field: str | None = None
) -> dict:
    prompt = f'''
Extrae los datos crediticios mencionados por el usuario.

Reglas:
- No inventes datos.
- Si un dato no aparece, devuélvelo como null.
- "No", "nunca" o "ninguno" representan 0 cuando se habla de
  incumplimientos, atrasos o marcas negativas.
- Devuelve únicamente JSON.

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
        'format': ExtractedApplicantData.model_json_schema(),
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

        return validated.model_dump(exclude_none=True)

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