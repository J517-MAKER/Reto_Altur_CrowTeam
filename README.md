# Altur Voice Detector — HackMTY 2026

Prototipo que clasifica llamadas telefónicas estéreo (8kHz, 16-bit PCM)
como realizadas por un **humano** o por **voz sintética (IA)**.

- Canal 0 = llamante a clasificar
- Canal 1 = agente del banco

## Estructura del proyecto

```
altur-voice-detector/
├── data/
│   ├── manifest.csv        # anon_id, label, split, duration_s
│   ├── audio/<anon_id>.wav
│   ├── turns/<anon_id>.json
│   └── features.csv        # generado por features.py
├── models/
│   └── model.joblib         # generado por train_model.py
├── features.py               # extracción de características (train + inferencia)
├── train_model.py             # entrenamiento del Random Forest
├── main.py                    # backend FastAPI (POST /detect)
├── dashboard.py                # dashboard Streamlit
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt 
```

Coloca tus datos reales en `data/manifest.csv`, `data/audio/` y `data/turns/`.

## Paso a paso

### 1. Extraer características

```bash
python features.py --manifest data/manifest.csv --audio-dir data/audio --turns-dir data/turns --out data/features.csv
```

Genera `data/features.csv` con ~50 columnas por llamada (energía, espectro,
pitch, MFCCs y métricas de toma de turnos), más `label` y `split`.

> Si `turns/<anon_id>.json` no existe o no coincide con el esquema esperado
> (ver comentarios en `features.py::load_turns`), el script usa un VAD
> simple como respaldo y avisa por consola — ajusta `load_turns()` si tu
> esquema real es distinto.

### 2. Entrenar el modelo

```bash
python train_model.py --features data/features.csv --model-out models/model.joblib
```

Entrena un `RandomForestClassifier` (con imputación de NaNs y balanceo de
clases), imprime `classification_report`, matriz de confusión, ROC-AUC y
las features más importantes sobre el split `val`, y guarda el pipeline
completo en `models/model.joblib`.

### 3. Levantar el backend

```bash
uvicorn main:app --reload --port 8000
```

Prueba rápida:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$(base64 -w0 data/audio/EJEMPLO.wav)\"}"
```

Respuesta esperada:

```json
{"is_synthetic": false, "confidence": 0.87}
```

### 4. Levantar el dashboard

```bash
streamlit run dashboard.py
```

Sube un WAV estéreo, visualiza ambos canales y consulta el endpoint
directamente desde la interfaz.

## Contrato Oficial del Juez (Evaluación en Vivo)

Durante la evaluación, el juez hace una petición `POST /detect` por cada llamada:

### Petición recibida por el endpoint (JSON)
```json
{
  "call_id": "call_4affad158a4c",
  "audio_base64": "<base64 del archivo WAV completo>",
  "sample_rate": 8000,
  "channels": 2
}
```
* **Canal 0:** Quien llama (audio del sujeto a clasificar).
* **Canal 1:** Agente del banco (contexto conversacional).
* **Tamaño:** Hasta ~5 MB (llamadas de 1 a 4 minutos).

### Respuesta requerida (HTTP 200, JSON)
```json
{
  "is_synthetic": true,
  "confidence": 0.87
}
```
* `is_synthetic` (booleano, **obligatorio**): `true` si es voz sintética/IA, `false` si es humano.
* `confidence` (float `0.0` - `1.0`, **opcional / recomendado**): Permite calcular ROC-AUC, calibración y sirve para desempatar.

### Reglas del Juez
* **Timeout:** Máximo **30 segundos** por llamada.
* **Criterio de fallo:** Timeout, status HTTP distinto de 200, o respuesta sin `is_synthetic` booleano cuenta como **incorrecta**.
* **Métrica principal:** **Balanced Accuracy** sobre un set de evaluación oculto con hablantes y voces inéditas.

---

## Scripts de Prueba Oficiales

Ambos scripts se encuentran en la carpeta `scripts/` y utilizan **únicamente la librería estándar de Python** (no requieren dependencias adicionales):

### 1. Probar tu endpoint con el simulador del juez (`check_endpoint.py`)

Envía llamadas reales del dataset a tu endpoint, valida el formato del esquema, mide latencias y reporta balanced accuracy y ROC-AUC:

```bash
python scripts/check_endpoint.py --url http://localhost:8000/detect --split val --n 20
```

Parámetros opcionales:
* `--url`: URL del endpoint (default: `http://localhost:8000/detect`)
* `--split`: Split del dataset a evaluar (`val`, `train`, `all`, default: `val`)
* `--n`: Número de llamadas a evaluar (default: `20`, usa `-1` o `0` para evaluar todas)
* `--timeout`: Límite en segundos por llamada (default: `30.0`)

### 2. Servidor de referencia mínimo (`example_server.py`)

Servidor ultraligero que implementa el contrato exacto del juez con una respuesta de prueba para verificar la tubería de punta a punta:

```bash
python scripts/example_server.py --port 8000
```

---

## Notas de diseño / arquitectura

- **Consistencia train/inferencia:** `extract_features_from_audio()` en `features.py` es la única función que genera el vector de features. En entrenamiento recibe los turnos del JSON; en inferencia (API) los turnos se estiman mediante VAD rápido.
- **Optimizaciones de latencia:** Modo `fast=True` (usa estimador YIN en lugar de pYIN) reduciendo el tiempo de procesamiento de ~30s a ~2-3s por llamada, manteniéndose muy por debajo del límite de 30 segundos del juez.
- **Caché LRU:** Caché en memoria por hash MD5 del audio para evitar reprocesamiento redundante.

