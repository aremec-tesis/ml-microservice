# Cognitive Performance API

Microservicio en Python con FastAPI que recomienda un ajuste de dificultad personalizado (`decrease` / `maintain` / `increase`) para pacientes con deterioro cognitivo a partir de metricas de sesion capturadas en una aplicacion de realidad virtual desarrollada en Unity.

Utiliza un modelo SVM (Support Vector Machine) **stateful** entrenado con scikit-learn y exportado a formato ONNX. El modelo consume tanto las 7 metricas de la sesion actual como **9 features agregadas del historial longitudinal del paciente** (ultimas 10 sesiones), produciendo directamente la recomendacion de dificultad junto con probabilidades por clase para trazabilidad clinica. Ademas se reporta un `cognitive_level` (low/medium/high) derivado deterministicamente del SPS como informacion adicional para el terapeuta.

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

## Generacion del dataset y entrenamiento

Antes de ejecutar el servidor por primera vez, generar el dataset sintetico longitudinal y entrenar el modelo:

```bash
# 1. Generar dataset (~6000 sesiones para 500 pacientes con trayectorias longitudinales)
python misc/generate_dataset.py

# 2. Entrenar el modelo SVM stateful (16 features, target = recommendation)
python misc/train.py
```

Esto genera:
- `misc/dataset/synthetic_vr_dataset.csv` — Dataset de entrenamiento
- `misc/model.onnx` — Modelo SVM stateful en formato ONNX (con `predict_proba`)
- `misc/scaler.joblib` — StandardScaler ajustado a las 16 features

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
  "correct_key_objects": 4,
  "correct_secondary_objects": 5,
  "incorrect_objects": 1,
  "total_key_objects": 5,
  "total_secondary_objects": 8,
  "total_events": 5,
  "correct_events": 4,
  "comprehension_score": 2,
  "response_times": [2.1, 3.5, 1.8, 2.0, 2.5, 1.9, 2.3, 2.1, 2.0, 1.7],
  "total_questions": 10,
  "incorrect_answers": 2,
  "interaction_events": 8,
  "expected_interactions": 10
}
```

Para la descripcion detallada de cada parametro, ver [REQUEST.md](REQUEST.md).

### Resumen de metricas y formulas

| Parametro(s) | Metrica | Formula |
|---|---|---|
| `correct_key_objects`, `correct_secondary_objects`, `incorrect_objects`, `total_key_objects`, `total_secondary_objects` | ORS | `((correct_key*2) + (correct_secondary*1) - (incorrect*1)) / ((total_key*2) + (total_secondary*1))` |
| `correct_events` / `total_events` | ERS | `correct_events / total_events` |
| `comprehension_score` | SCS | `comprehension_score / 2` |
| `response_times` | RTA | `mean(response_times)` |
| `interaction_events` / `expected_interactions` | ATS | `interaction_events / expected_interactions` |
| `incorrect_answers` / `total_questions` | ER | `incorrect_answers / total_questions` |
| ORS, ERS, SCS, ER | SPS | `0.3*ORS + 0.3*ERS + 0.2*SCS + 0.2*(1-ER)` |

> **Nota sobre ORS**: la nueva formula pondera doble los objetos clave sobre los secundarios y resta los objetos incorrectamente identificados. Puede tomar valores negativos cuando los errores superan los aciertos ponderados.

### Response

```json
{
  "metrics": {
    "ors": 0.667,
    "ers": 0.8,
    "scs": 1.0,
    "rta": 2.19,
    "ats": 0.8,
    "er": 0.2,
    "sps": 0.8
  },
  "cognitive_level": "high",
  "recommendation": "maintain_difficulty",
  "probabilities": {
    "decrease_difficulty": 0.000,
    "maintain_difficulty": 0.992,
    "increase_difficulty": 0.008
  },
  "context": {
    "baseline_sps": 0.582,
    "slope_sps": -0.05,
    "delta_sps": 0.218,
    "mean_ors": 0.65,
    "mean_ers": 0.55,
    "mean_er": 0.25,
    "mean_rta": 2.333,
    "std_sps": 0.041,
    "session_count": 3,
    "cold_start": false
  }
}
```

| Campo | Descripcion |
|---|---|
| `metrics` | Las 7 metricas cognitivas calculadas para la sesion actual |
| `cognitive_level` | Nivel cognitivo derivado deterministicamente del SPS (`<0.4` low, `0.4-0.7` medium, `>0.7` high). Informativo |
| `recommendation` | Recomendacion de dificultad producida por el ML stateful |
| `probabilities` | Confianza del modelo en cada clase (suma 1.0). Permite trazabilidad clinica |
| `context` | Las 9 features de historial que el ML consumio + flag `cold_start` |

### Las 16 features del modelo

El modelo SVM stateful consume un vector de 16 features:

| # | Feature | Origen |
|---|---------|--------|
| 1-7 | ORS, ERS, SCS, RTA, ATS, ER, SPS | Sesion actual |
| 8 | `baseline_sps` | Media movil ponderada del SPS sobre las ultimas 10 sesiones |
| 9 | `slope_sps` | Pendiente lineal del SPS (positivo = mejora, negativo = declive) |
| 10 | `delta_sps` | `SPS_actual - baseline_sps` |
| 11-14 | `mean_ors`, `mean_ers`, `mean_er`, `mean_rta` | Promedios historicos |
| 15 | `std_sps` | Desviacion estandar del SPS historico |
| 16 | `session_count` | Numero de sesiones previas (0 en cold start) |

En **cold start** (primera sesion del paciente), las features de historial se neutralizan: `baseline_sps = SPS_actual`, `slope=0`, `delta=0`, `std=0`, `mean_*` igualan los valores actuales, `session_count=0`.

### Trazabilidad clinica

Cada inferencia persiste en `schema_telemetria.metricas_sesion`:
- Las metricas raw enviadas
- Las 7 metricas cognitivas calculadas
- Las 9 features de historial usadas como entrada al modelo
- La recomendacion del modelo y las 3 probabilidades por clase
- El `cognitive_level` derivado del SPS

Cualquier prediccion puede reconstruirse y explicarse a partir de la base de datos sin necesidad de un motor de reglas externo.

### Mock requests para testing

**Caso 1 — Rendimiento alto, cold start** (esperado: `cognitive_level: high`, recomendacion conservadora `maintain` o `increase` segun el modelo)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 101,
    "correct_key_objects": 5,
    "correct_secondary_objects": 4,
    "incorrect_objects": 0,
    "total_key_objects": 5,
    "total_secondary_objects": 5,
    "total_events": 10,
    "correct_events": 8,
    "comprehension_score": 2,
    "response_times": [1.5, 2.0, 1.8, 1.6, 1.7, 1.9, 1.5, 2.1],
    "total_questions": 10,
    "incorrect_answers": 1,
    "interaction_events": 9,
    "expected_interactions": 10
  }'
```

**Caso 2 — Rendimiento medio, cold start** (esperado: `cognitive_level: medium`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 102,
    "correct_key_objects": 3,
    "correct_secondary_objects": 3,
    "incorrect_objects": 1,
    "total_key_objects": 5,
    "total_secondary_objects": 5,
    "total_events": 10,
    "correct_events": 5,
    "comprehension_score": 1,
    "response_times": [3.5, 4.2, 3.8, 4.5, 4.0, 3.9],
    "total_questions": 10,
    "incorrect_answers": 3,
    "interaction_events": 6,
    "expected_interactions": 10
  }'
```

**Caso 3 — Rendimiento bajo, cold start** (esperado: `cognitive_level: low`, recomendacion `decrease`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 103,
    "correct_key_objects": 1,
    "correct_secondary_objects": 1,
    "incorrect_objects": 3,
    "total_key_objects": 5,
    "total_secondary_objects": 5,
    "total_events": 10,
    "correct_events": 2,
    "comprehension_score": 0,
    "response_times": [6.0, 7.5, 8.0, 6.8, 7.2, 6.5],
    "total_questions": 10,
    "incorrect_answers": 8,
    "interaction_events": 2,
    "expected_interactions": 10
  }'
```

> Para observar la influencia del historial, enviar multiples requests con el mismo `patient_id` simulando una trayectoria temporal. A partir del segundo request `cold_start` sera `false` y las features de historial se calcularan con sesiones reales, modulando la recomendacion del modelo.

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
    public int correct_key_objects;
    public int correct_secondary_objects;
    public int incorrect_objects;
    public int total_key_objects;
    public int total_secondary_objects;
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
public class Probabilities
{
    public float decrease_difficulty;
    public float maintain_difficulty;
    public float increase_difficulty;
}

[System.Serializable]
public class PatientContext
{
    public float baseline_sps;
    public float slope_sps;
    public float delta_sps;
    public float mean_ors;
    public float mean_ers;
    public float mean_er;
    public float mean_rta;
    public float std_sps;
    public int session_count;
    public bool cold_start;
}

[System.Serializable]
public class PredictionResponse
{
    public SessionMetrics metrics;
    public string cognitive_level;
    public string recommendation;
    public Probabilities probabilities;
    public PatientContext context;
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
            Debug.Log($"Recommendation: {response.recommendation}");
            Debug.Log($"Cognitive level: {response.cognitive_level}");
            Debug.Log($"SPS: {response.metrics.sps}");
            Debug.Log($"Confidence: dec={response.probabilities.decrease_difficulty} " +
                      $"maint={response.probabilities.maintain_difficulty} " +
                      $"inc={response.probabilities.increase_difficulty}");
            Debug.Log($"Sessions tracked: {response.context.session_count}");
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
├── main.py                          # Entry point (shim que arranca interfaces/http/api.py)
├── requirements.txt                 # Dependencias
├── .env                             # Credenciales de base de datos (no versionado)
│
├── domain/                          # Tipos de dominio puros, sin dependencias de framework
│   ├── session_metrics.py           # RawSessionData, SessionMetrics, CognitiveLevel, cognitive_level_from_sps
│   ├── patient_context.py           # HistoricalSession, PatientContext, feature_vector (16 features)
│   └── difficulty_recommendation.py # DifficultyRecommendation enum
│
├── app/                             # Orquestacion CQRS
│   ├── commands/
│   │   └── predict_session.py       # PredictSessionCommand + Handler (flujo completo)
│   └── queries/
│       └── get_patient_history.py   # GetPatientHistoryQuery + Handler
│
├── infrastructure/                  # Adaptadores tecnicos
│   ├── config.py                    # Paths, settings, HISTORY_WINDOW
│   ├── ml/
│   │   └── onnx_classifier.py       # Wrapper SVM stateful ONNX + scaler (label + probs)
│   └── persistence/
│       ├── postgres_pool.py         # Pool asyncpg + DDL
│       └── session_repository.py    # insert_session, get_patient_history
│
├── interfaces/                      # Puntos de entrada HTTP
│   └── http/
│       ├── api.py                   # FastAPI app + lifespan + endpoint /predict
│       └── schemas.py               # Pydantic DTOs de entrada y salida
│
└── misc/                            # Artefactos de ML y herramientas
    ├── generate_dataset.py          # Generador de dataset sintetico longitudinal
    ├── train.py                     # Script de entrenamiento y exportacion a ONNX
    ├── model.onnx                   # Modelo SVM exportado (generado por train.py)
    ├── scaler.joblib                # StandardScaler ajustado (generado por train.py)
    └── dataset/
        └── synthetic_vr_dataset.csv # Dataset de entrenamiento (generado por generate_dataset.py)
```

## Tecnologias

- **FastAPI** — Framework web asincrono
- **ONNX Runtime** — Inferencia del modelo SVM
- **scikit-learn** — Entrenamiento del modelo SVM (con `probability=True`)
- **skl2onnx** — Conversion de sklearn a ONNX (con `zipmap=False` para exponer probabilidades)
- **asyncpg** — Cliente asincrono para PostgreSQL
- **Pydantic** — Validacion de datos de entrada/salida
- **python-dotenv** — Carga de variables de entorno
- **uvicorn** — Servidor ASGI

## Base de datos

El microservicio persiste cada sesion en el esquema `schema_telemetria` de PostgreSQL (TimescaleDB). La tabla `metricas_sesion` se crea automaticamente al iniciar el servidor e incluye:

- Datos raw enviados (incluyendo objetos clave/secundario/incorrectos)
- Las 7 metricas cognitivas calculadas
- Las 9 features de historial que el ML consumio (`baseline_sps`, `slope_sps`, `delta_sps`, `mean_ors`, `mean_ers`, `mean_er`, `mean_rta`, `std_sps`, `session_count`, `cold_start`)
- La recomendacion del modelo y las 3 probabilidades por clase (`prob_decrease`, `prob_maintain`, `prob_increase`)
- El `cognitive_level` derivado del SPS

Esto garantiza trazabilidad clinica completa: cualquier decision del modelo es reconstruible y explicable a partir de la base de datos. Ver `infrastructure/persistence/postgres_pool.py` para el DDL completo.
