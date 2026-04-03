# Cognitive Performance API

Microservicio en Python con FastAPI que predice el nivel de rendimiento cognitivo de pacientes con deterioro cognitivo a partir de metricas de sesion capturadas en una aplicacion de realidad virtual desarrollada en Unity.

Utiliza un modelo SVM (Support Vector Machine) entrenado con scikit-learn, exportado a formato ONNX para inferencia de alto rendimiento. Clasifica el rendimiento en tres niveles: **low**, **medium** y **high**, y retorna una recomendacion de ajuste de dificultad para la siguiente sesion.

## Requisitos

- Python 3.10 o superior
- PostgreSQL con TimescaleDB (para persistencia de metricas)

## Instalacion

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Configuracion

Crear un archivo `.env` en la raiz del proyecto con las credenciales de la base de datos:

```env
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=postgres
DATABASE_PASSWORD=changeme
DATABASE_NAME=aremec
```

## Entrenamiento del modelo

Antes de ejecutar el servidor por primera vez, se debe entrenar el modelo:

```bash
python train.py
```

Esto genera dos archivos:
- `model.onnx` — Modelo SVM en formato ONNX
- `scaler.joblib` — StandardScaler ajustado al dataset

El script utiliza el dataset en `dataset/synthetic_vr_dataset.csv` (2149 muestras, 3 clases).

## Ejecucion

```bash
python main.py
```

El servidor se inicia en `http://127.0.0.1:8000`. Al arrancar, carga el modelo ONNX y el scaler en memoria, e inicializa el pool de conexiones a PostgreSQL.

## Uso

### Endpoint

**POST** `/predict`

### Request

```json
{
  "patient_id": 1,
  "total_objects": 10,
  "correct_objects": 7,
  "total_events": 5,
  "correct_events": 4,
  "comprehension_score": 2,
  "response_times": [2.1, 3.5, 1.8],
  "total_questions": 10,
  "incorrect_answers": 2,
  "interaction_events": 8,
  "expected_interactions": 10
}
```

| Campo | Tipo | Validacion | Descripcion |
|---|---|---|---|
| `patient_id` | int | — | ID anonimizado del paciente |
| `total_objects` | int | >= 0 | Objetos totales en la escena |
| `correct_objects` | int | >= 0 | Objetos recordados correctamente |
| `total_events` | int | >= 0 | Eventos totales en la narrativa |
| `correct_events` | int | >= 0 | Eventos recordados correctamente |
| `comprehension_score` | int | 0-2 | Puntuacion de comprension narrativa |
| `response_times` | float[] | — | Tiempos de respuesta en segundos |
| `total_questions` | int | >= 0 | Preguntas totales realizadas |
| `incorrect_answers` | int | >= 0 | Respuestas incorrectas |
| `interaction_events` | int | >= 0 | Eventos de interaccion registrados |
| `expected_interactions` | int | >= 0 | Interacciones esperadas |

### Response

```json
{
  "metrics": {
    "ors": 0.7,
    "ers": 0.8,
    "scs": 1.0,
    "rta": 2.4667,
    "ats": 0.8,
    "er": 0.2,
    "sps": 0.81
  },
  "prediction": "high",
  "recommendation": "increase_difficulty"
}
```

### Metricas cognitivas calculadas

| Metrica | Nombre | Formula |
|---|---|---|
| ORS | Object Recall Score | `correct_objects / total_objects` |
| ERS | Event Recall Score | `correct_events / total_events` |
| SCS | Story Comprehension Score | `comprehension_score / 2` |
| RTA | Response Time Average | `mean(response_times)` |
| ATS | Attention Score | `interaction_events / expected_interactions` |
| ER | Error Rate | `incorrect_answers / total_questions` |
| SPS | Session Performance Score | `0.3*ORS + 0.3*ERS + 0.2*SCS + 0.2*(1-ER)` |

### Recomendacion de dificultad

| Condicion | Recomendacion |
|---|---|
| SPS < 0.4 | `decrease_difficulty` |
| 0.4 ≤ SPS ≤ 0.7 | `maintain_difficulty` |
| SPS > 0.7 | `increase_difficulty` |

### Ejemplo con curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 1,
    "total_objects": 10,
    "correct_objects": 7,
    "total_events": 5,
    "correct_events": 4,
    "comprehension_score": 2,
    "response_times": [2.1, 3.5, 1.8],
    "total_questions": 10,
    "incorrect_answers": 2,
    "interaction_events": 8,
    "expected_interactions": 10
  }'
```

### Ejemplo en Unity (C#)

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System.Collections;

[System.Serializable]
public class SessionInput
{
    public int patient_id;
    public int total_objects;
    public int correct_objects;
    public int total_events;
    public int correct_events;
    public int comprehension_score;
    public float[] response_times;
    public int total_questions;
    public int incorrect_answers;
    public int interaction_events;
    public int expected_interactions;
}

[System.Serializable]
public class SessionMetrics
{
    public float ors;
    public float ers;
    public float scs;
    public float rta;
    public float ats;
    public float er;
    public float sps;
}

[System.Serializable]
public class PredictionResponse
{
    public SessionMetrics metrics;
    public string prediction;
    public string recommendation;
}

public class CognitiveAPI : MonoBehaviour
{
    private string apiUrl = "http://localhost:8000/predict";

    public IEnumerator SendPrediction(SessionInput input)
    {
        string json = JsonUtility.ToJson(input);
        byte[] body = Encoding.UTF8.GetBytes(json);

        using var request = new UnityWebRequest(apiUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(body);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            var response = JsonUtility.FromJson<PredictionResponse>(
                request.downloadHandler.text
            );
            Debug.Log($"Prediction: {response.prediction}");
            Debug.Log($"Recommendation: {response.recommendation}");
            Debug.Log($"SPS: {response.metrics.sps}");
        }
        else
        {
            Debug.LogError($"Error: {request.error}");
        }
    }
}
```

## Documentacion interactiva

Con el servidor en ejecucion, accede a la documentacion generada automaticamente por FastAPI:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Estructura del proyecto

```
ml_microservice/
├── main.py              # Aplicacion FastAPI (lifespan, CORS, endpoint)
├── metrics.py           # Modelos Pydantic, calculo de metricas, recomendacion
├── model_handler.py     # Carga de modelo ONNX y scaler, inferencia
├── database.py          # Pool asyncpg, queries a PostgreSQL
├── train.py             # Script de entrenamiento y exportacion a ONNX
├── requirements.txt     # Dependencias
├── .env                 # Credenciales de base de datos (no versionado)
├── dataset/
│   └── synthetic_vr_dataset.csv  # Dataset de entrenamiento
├── model.onnx           # Modelo SVM exportado (generado por train.py)
├── scaler.joblib        # StandardScaler ajustado (generado por train.py)
└── README.md
```

## Tecnologias

- **FastAPI** — Framework web asincrono
- **ONNX Runtime** — Inferencia del modelo SVM
- **scikit-learn** — Entrenamiento del modelo SVM
- **skl2onnx** — Conversion de sklearn a ONNX
- **asyncpg** — Cliente asincrono para PostgreSQL
- **Pydantic** — Validacion de datos de entrada/salida
- **python-dotenv** — Carga de variables de entorno
- **uvicorn** — Servidor ASGI

## Base de datos

El microservicio persiste cada sesion en el esquema `schema_telemetria` de PostgreSQL (TimescaleDB). La tabla `metricas_sesion` se crea automaticamente al iniciar el servidor. Consultar el archivo `database.py` para el DDL completo.
