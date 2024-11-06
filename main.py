import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from transformers import pipeline
import sqlite3
import requests

# Configuración de la aplicación FastAPI
app = FastAPI()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración del clasificador de sentimiento
clasificador = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment', framework='pt')

# Configuración de la base de datos SQLite
conn = sqlite3.connect('historial_analisis.db', check_same_thread=False)
cursor = conn.cursor()

# Crear tabla de historial si no existe
cursor.execute('''
    CREATE TABLE IF NOT EXISTS historial (
        id INTEGER PRIMARY KEY,
        texto TEXT,
        clasificacion TEXT,
        sentimiento INTEGER
    )
''')
conn.commit()

# Umbrales de sentimiento
UMBRAL_POSITIVO = 3
UMBRAL_NEGATIVO = 2

# Configuración de templates y archivos estáticos
templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mapeo de clasificaciones a valores numéricos
mapeo_clasificaciones = {
    '5 stars': 5,
    '4 stars': 4,
    '3 stars': 3,
    '2 stars': 2,
    '1 star': 1
}

# Función para filtrar publicaciones según el sentimiento
def filtro_publicacion(sentimiento: float) -> bool:
    return sentimiento >= UMBRAL_POSITIVO

@app.get("/")
async def read_index():
    return HTMLResponse(content=open("static/index.html", "rb").read(), media_type="text/html")

@app.get("/analisisTexto")
def analizar_sentimiento_con_filtro(texto: str):
    logger.info(f"Texto recibido para análisis: {texto}")
    
    # Análisis de sentimiento
    resultado = clasificador(texto)
    logger.info(f"Resultado del análisis de sentimientos: {resultado}")
    clasificacion = resultado[0]['label']
    sentimiento = mapeo_clasificaciones.get(clasificacion, "Desconocido")
    
    # Normalización del sentimiento
    if sentimiento in [1, 2]:
        sentimiento = 1
    elif sentimiento == 3:
        sentimiento = 2
    else:
        sentimiento = 3

    cumple_filtro = filtro_publicacion(sentimiento)
    mensaje = generar_mensaje_individual(sentimiento)
    
    # Guardar en la base de datos
    guardar_historial(texto, clasificacion, sentimiento)
    
    return {
        "texto": texto,
        "clasificacion": clasificacion,
        "sentimiento": sentimiento,
        "cumple_filtro": cumple_filtro,
        "mensaje": mensaje,
        "promedio_sentimiento": calcular_promedio_sentimiento()
    }

@app.get("/analizarComentariosReddit")
def analizar_comentarios_reddit_con_filtro(subreddit: str, limit: int = 10):
    logger.info(f"Subreddit recibido para análisis: {subreddit}")
    url = f"https://www.reddit.com/r/{subreddit}/comments.json?limit={limit}"
    headers = {'User-agent': 'sentiment-analyzer'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lanza un error si la solicitud falla
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al obtener los comentarios de Reddit: {e}")
        raise HTTPException(status_code=500, detail="Error al obtener los comentarios de Reddit")
    
    try:
        comentarios = response.json()['data']['children']
    except KeyError:
        logger.error("La respuesta de Reddit no tiene el formato esperado.")
        raise HTTPException(status_code=500, detail="Error al procesar los comentarios de Reddit")
    
    resultados_comentarios = []
    
    for comentario in comentarios:
        texto = comentario['data']['body']
        resultado = clasificador(texto)
        logger.info(f"Comentario: {texto}, Resultado del análisis de sentimientos: {resultado}")
        
        clasificacion = resultado[0]['label']
        sentimiento = mapeo_clasificaciones.get(clasificacion, "Desconocido")
        
        # Normalización del sentimiento
        if sentimiento in [1, 2]:
            sentimiento = 1
        elif sentimiento == 3:
            sentimiento = 2
        else:
            sentimiento = 3
        
        cumple_filtro = filtro_publicacion(sentimiento)
        mensaje = generar_mensaje_individual(sentimiento)
        
        # Guardar en la base de datos
        guardar_historial(texto, clasificacion, sentimiento)
        
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
        "mensaje": mensaje,
        "promedio_sentimiento": calcular_promedio_sentimiento()
    }

@app.get("/historial")
async def ver_historial(request: Request):
    cursor.execute("SELECT texto, clasificacion, sentimiento FROM historial")
    historial = cursor.fetchall()
    return templates.TemplateResponse("historial.html", {"request": request, "historial": historial})

# Generar mensaje para comentarios de Reddit
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

# Generar mensaje individual para análisis de texto
def generar_mensaje_individual(sentimiento: float) -> str:
    if sentimiento >= 3:
        return "El análisis de sentimientos indica una gran positividad en el texto."
    elif sentimiento == 2:
        return "El análisis de sentimientos revela una neutralidad en el texto."
    else:
        return "El análisis de sentimientos detecta un tono negativo en el texto."

# Calcular promedio de sentimiento en el historial
def calcular_promedio_sentimiento() -> float:
    cursor.execute("SELECT AVG(sentimiento) FROM historial")
    promedio = cursor.fetchone()[0]
    return promedio if promedio else 0.0

# Guardar análisis en historial
def guardar_historial(texto: str, clasificacion: str, sentimiento: int):
    cursor.execute("INSERT INTO historial (texto, clasificacion, sentimiento) VALUES (?, ?, ?)", 
                   (texto, clasificacion, sentimiento))
    conn.commit()

@app.get("/reiniciarHistorial")
def reiniciar_historial():
    try:
        cursor.execute("DELETE FROM historial")  # Elimina todos los registros de la tabla
        conn.commit()  # Confirma los cambios en la base de datos
        return {"mensaje": "Historial reiniciado con éxito."}
    except Exception as e:
        conn.rollback()  # Si hay un error, deshace los cambios
        logger.error(f"Error al reiniciar el historial: {e}")
        return {"mensaje": "Hubo un error al reiniciar el historial.", "error": str(e)}