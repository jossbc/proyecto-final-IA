import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.getenv('MONGODB_URI')
    if not uri:
        raise RuntimeError(
            'No se encontró la variable MONGODB_URI.'
        )

    return MongoClient(
        uri,
        serverSelectionTimeoutMS=5000
    )


def get_collection():
    database_name = os.getenv('DATABASE_NAME')
    collection_name = os.getenv("COLLECTION_NAME")
    
    if not database_name:
        raise RuntimeError(
            'No se encontró la variable DATABASE_NAME.'
        )

    if not collection_name:
        raise RuntimeError(
            'No se encontró la variable COLLECTION_NAME.'
        )

    return get_client()[database_name][collection_name]
