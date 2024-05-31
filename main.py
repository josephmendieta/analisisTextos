from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/face")
def face_detect():
    return {"servici": "Detección de rostros"}
