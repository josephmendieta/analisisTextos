# Proyecto de Análisis de Sentimientos

Este proyecto consiste en una aplicación web para el análisis de sentimientos utilizando FastAPI y el modelo preentrenado `nlptown/bert-base-multilingual-uncased-sentiment` de Hugging Face Transformers. La aplicación evalúa el sentimiento de texto ingresado y comentarios extraídos de Reddit, clasificándolos como positivos, negativos o neutrales.

## Características

- Análisis de sentimientos de textos ingresados.
- Recuperación y análisis de comentarios de Reddit.
- Filtrado de resultados basado en umbrales de sentimiento.

## Instalación

1. Clona este repositorio en tu máquina local:

   ```bash
   git clone https://github.com/josephmendieta/analisisTextos.git
   cd analisisTextos
   
2. Instala las dependencias del proyecto utilizando el archivo requirements.txt:
```bash
pip install -r requirements.txt
```

3. En caso de aún generar errores, en la terminal del equipo cliente ejecutar:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

## Asegúrate de tener instalado PyTorch o TensorFlow según prefieras:

Para instalar PyTorch, sigue las instrucciones en pytorch.org.
Para instalar TensorFlow, sigue las instrucciones en tensorflow.org.

## Uso

Ejecuta la aplicación con el siguiente comando: 

```bash
python -m uvicorn main:app --reload
```
 
Accede a la interfaz de usuario en tu navegador web ingresando la dirección http://localhost:8000. La documentación interactiva de la API está disponible en http://localhost:8000/docs.

## Créditos
- FastAPI: Documentación
- Modelo preentrenado de Hugging Face Transformers: nlptown/bert-base-multilingual-uncased-sentiment
- Google Colab para pruebas y desarrollo: Colab Notebook

## Contribuciones
Las contribuciones son bienvenidas. Si deseas mejorar este proyecto, no dudes en enviar un pull request.

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para obtener más detalles.
