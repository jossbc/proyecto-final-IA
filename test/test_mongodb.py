from utils.mongo import client, collection

try:
    client.admin.command("ping")
    print("Conexión exitosa a MongoDB")

    result = collection.insert_one({
        "test": True,
        "message": "Hola MongoDB"
    })

    print("Documento insertado:", result.inserted_id)

except Exception as e:
    print("Error:", e)