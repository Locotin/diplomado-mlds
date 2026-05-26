# Emotion Stack

Proyecto base para una arquitectura de deteccion de emociones con tres servicios:

- `input`: recibe imagenes por HTTP y delega el procesamiento.
- `ingesta`: sube la imagen a Cloudinary, invoca el modelo y persiste resultados en MongoDB Atlas.
- `modelo`: expone inferencia HTTP usando un modelo real de Hugging Face.

El dashboard no vive como contenedor. La exploracion y los diagramas se hacen desde notebooks en `notebooks/colab`, pensados para Google Colab o para ejecucion local.

## Estructura

```text
BD/Proyecto/
├── docker-compose.yml
├── notebooks/colab
├── services
│   ├── input
│   ├── ingesta
│   └── modelo
└── data/input
```

## Variables de entorno

1. Copia `.env.example` a `.env` si necesitas reiniciar la configuracion.
2. Completa obligatoriamente:
   - `MONGODB_URI`
   - `CLOUDINARY_CLOUD_NAME`
   - `CLOUDINARY_API_KEY`
   - `CLOUDINARY_API_SECRET`

Variables importantes:

- `MODEL_ID`: por defecto `abhilash88/face-emotion-detection`
- `MODEL_DEVICE`: `cpu`, `cuda`, `mps` o `auto`
- `MODEL_TOP_K`: cantidad de emociones a devolver

## Levantar el stack

```bash
docker compose up --build
```

Servicios expuestos:

- `http://localhost:8002` -> `input`
- `http://localhost:8001` -> `ingesta`
- `http://localhost:8000` -> `modelo`

## Flujo principal

1. Enviar una imagen a `POST /uploads` del servicio `input`.
2. `input` reenvia el archivo a `ingesta`.
3. `ingesta` sube la imagen a Cloudinary.
4. `ingesta` llama `POST /infer` de `modelo` con la URL de Cloudinary.
5. `ingesta` persiste el resultado en MongoDB Atlas y devuelve el documento consolidado.

## Ejemplo de subida

```bash
curl -X POST http://localhost:8002/uploads \
  -F "file=@/ruta/a/imagen.jpg" \
  -F 'metadata_json={"cohort":"demo","source":"manual"}'
```

## Notebook de dashboard

`notebooks/colab/emotion-dashboard-colab.ipynb` consulta MongoDB Atlas directamente y genera diagramas con Plotly.

Notas para Google Colab:

- Colab no puede alcanzar `localhost` de tu maquina. Para consumir la API desde Colab necesitas exponerla con un tunel.
- El notebook ya soporta el camino mas directo: consultar Atlas con `MONGODB_URI`.

## Validacion local sin Docker

Compilacion rapida de Python:

```bash
python3 -m compileall services notebooks/colab
```

Pruebas unitarias:

```bash
pytest tests
```
