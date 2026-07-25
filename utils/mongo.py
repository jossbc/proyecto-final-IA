from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGODB_URI")

client = MongoClient(uri)

db = client[os.getenv("DATABASE_NAME")]

collection = db[os.getenv("COLLECTION_NAME")]