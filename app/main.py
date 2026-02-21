from fastapi import FastAPI

app = FastAPI(title="API Básica")


@app.get("/")
def funciona():
    """Endpoint GET que confirma que la API funciona."""
    return {"mensaje": "Funciona"}
