import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    mongodb_uri: str
    mongodb_database: str
    llm_api_key: str
    llm_model: str

    def __init__(self) -> None:
        self.mongodb_uri = self._required("MONGODB_URI")
        self.mongodb_database = os.getenv(
            "MONGODB_DATABASE",
            "credit_risk_db",
        )
        self.llm_api_key = self._required("LLM_API_KEY")
        self.llm_model = os.getenv(
            "LLM_MODEL",
            "default-model",
        )

    @staticmethod
    def _required(variable_name: str) -> str:
        value = os.getenv(variable_name)

        if not value:
            raise RuntimeError(
                f"No se encontró la variable de entorno {variable_name}."
            )

        return value


settings = Settings()