import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from transformers import pipeline
import requests

app = FastAPI()

# Configurar el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

clasificador = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

UMBRAL_POSITIVO = 3
UMBRAL_NEGATIVO = 2

app.mount("/static", StaticFiles(directory="static"), name="static")

mapeo_clasificaciones = {
    '5 stars': 5,
    '4 stars': 4,
    '3 stars': 3,
    '2 stars': 2,
    '1 star': 1
}

def filtro_publicacion(sentimiento: float) -> bool:
    return sentimiento >= UMBRAL_POSITIVO

@app.get("/")
async def read_index():
    return StreamingResponse(open("static/index.html", "rb"), media_type="text/html")

@app.get("/analisisTexto")
def analizar_sentimiento_con_filtro(texto: str):
    logger.info(f"Texto recibido para análisis: {texto}")
    resultado = clasificador(texto)
    logger.info(f"Resultado del análisis de sentimientos: {resultado}")
    clasificacion = resultado[0]['label']
    sentimiento = mapeo_clasificaciones.get(clasificacion, "Desconocido")
    if sentimiento in [1, 2]:
        sentimiento = 1
    elif sentimiento == 3:
        sentimiento = 2
    else:
        sentimiento = 3
    cumple_filtro = filtro_publicacion(sentimiento)
    mensaje = generar_mensaje_individual(sentimiento)
    return {
        "texto": texto,
        "clasificacion": clasificacion,
        "sentimiento": sentimiento,
        "cumple_filtro": cumple_filtro,
        "mensaje": mensaje
    }

@app.get("/analizarComentariosReddit")
def analizar_comentarios_reddit_con_filtro(subreddit: str, limit: int = 10):
    logger.info(f"Subreddit recibido para análisis: {subreddit}")
    url = f"https://www.reddit.com/r/{subreddit}/comments.json?limit={limit}"
    headers = {'User-agent': 'sentiment-analyzer'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error al obtener los comentarios de Reddit")
    
    comentarios = response.json()['data']['children']
    resultados_comentarios = []
    
    for comentario in comentarios:
        texto = comentario['data']['body']
        resultado = clasificador(texto)
        logger.info(f"Comentario: {texto}, Resultado del análisis de sentimientos: {resultado}")
        clasificacion = resultado[0]['label']
        sentimiento = mapeo_clasificaciones.get(clasificacion, "Desconocido")
        if sentimiento in [1, 2]:
            sentimiento = 1
        elif sentimiento == 3:
            sentimiento = 2
        else:
            sentimiento = 3
        cumple_filtro = filtro_publicacion(sentimiento)
        mensaje = generar_mensaje_individual(sentimiento)
        resultados_comentarios.append({
            "texto": texto,
            "clasificacion": clasificacion,
            "sentimiento": sentimiento,
            "cumple_filtro": cumple_filtro,
            "mensaje": mensaje
        })
    
    mensaje = generar_mensaje_reddit(resultados_comentarios)
    return {
        "subreddit": subreddit,
        "comentarios_analizados": resultados_comentarios,
        "mensaje": mensaje
    }

def generar_mensaje_reddit(comentarios: list) -> str:
    if not comentarios:
        return "No se encontraron comentarios que cumplan con los criterios de filtro."
    
    comentarios_filtrados = [comentario for comentario in comentarios if comentario['cumple_filtro']]
    if not comentarios_filtrados:
        return "No hay comentarios que pasen el filtro."
    
    clasificaciones_numericas = [comentario['sentimiento'] for comentario in comentarios_filtrados]
    clasificacion_promedio = sum(clasificaciones_numericas) / len(clasificaciones_numericas)

    if clasificacion_promedio >= 4.0:
        mensaje = "¡Los comentarios de este tema en Reddit son muy positivos!"
    elif 2.0 <= clasificacion_promedio < 4.0:
        mensaje = "Los comentarios de este tema en Reddit reflejan opiniones mixtas."
    else:
        mensaje = "Los comentarios de este tema en Reddit son principalmente negativos o críticos."
    return mensaje

def generar_mensaje_individual(sentimiento: float) -> str:
    if sentimiento >= 3:
        mensaje = "El análisis de sentimientos indica una gran positividad en el texto."
    elif sentimiento == 2:
        mensaje = "El análisis de sentimientos revela una neutralidad en el texto."
    else:
        mensaje = "El análisis de sentimientos detecta un tono negativo en el texto."
    return mensaje
