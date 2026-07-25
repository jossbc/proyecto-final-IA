from utils.mongo import client, collection

try:
    # Verifica que la conexión funcione
    client.admin.command("ping")
    print("Conexión exitosa a MongoDB")

    # Inserta un documento de prueba
    result = collection.insert_one({
        "test": True,
        "message": "Hola MongoDB"
    })

    print("Documento insertado:", result.inserted_id)

except Exception as e:
    print("Error:", e)