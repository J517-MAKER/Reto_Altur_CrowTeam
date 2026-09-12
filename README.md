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

## Notas de diseño / próximos pasos

- **Consistencia train/inferencia:** `extract_features_from_audio()` en
  `features.py` es la única función que genera el vector de features.
  En entrenamiento recibe los turnos reales del JSON; en inferencia (API)
  no hay JSON disponible, así que los turnos se estiman con un VAD basado
  en `librosa.effects.split`. Si el reto permite mandar turnos reales en
  producción, es fácil extenderlo.
- **Mejoras razonables para siguientes iteraciones:** embeddings
  pre-entrenados (p.ej. wav2vec2/x-vectors) en vez de features hechas a
  mano, calibración de probabilidades, y detección de artefactos típicos
  de vocoders (armónicos residuales, fase).
