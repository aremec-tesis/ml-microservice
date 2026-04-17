# Cognitive Performance API

Microservicio en Python con FastAPI que predice el nivel de rendimiento cognitivo de pacientes con deterioro cognitivo a partir de metricas de sesion capturadas en una aplicacion de realidad virtual desarrollada en Unity.

Utiliza un modelo SVM (Support Vector Machine) entrenado con scikit-learn, exportado a formato ONNX para inferencia de alto rendimiento. Clasifica el rendimiento en tres niveles: **low**, **medium** y **high**, y retorna una recomendacion de ajuste de dificultad **personalizada al historial longitudinal del paciente**.

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
python misc/train.py
```

Esto genera dos archivos dentro de `misc/`:
- `misc/model.onnx` — Modelo SVM en formato ONNX
- `misc/scaler.joblib` — StandardScaler ajustado al dataset

El script utiliza el dataset en `misc/dataset/synthetic_vr_dataset.csv`.

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

### Descripcion de parametros

---

**`patient_id`** · `int` · Requerido

Identificador numerico unico del paciente dentro del sistema. Se usa para recuperar el historial de sesiones anteriores y personalizar la recomendacion de dificultad. No se valida su existencia previa — si es la primera sesion del paciente, el sistema opera en modo cold start usando unicamente los datos de la sesion actual.

---

**`total_objects`** · `int >= 0` · Requerido

Cantidad total de objetos que fueron presentados al paciente durante la escena VR para que los memorice o reconozca. Actua como denominador en el calculo de ORS (Object Recall Score). Debe ser mayor que cero para que la metrica tenga valor; si se envia 0, ORS se fija en 0.

---

**`correct_objects`** · `int >= 0` · Requerido

Cantidad de objetos que el paciente identifico o recordo correctamente al ser evaluado. Debe ser menor o igual a `total_objects`. Se usa junto con `total_objects` para calcular ORS = `correct_objects / total_objects`.

---

**`total_events`** · `int >= 0` · Requerido

Cantidad total de eventos narrativos que ocurrieron durante la sesion VR (acciones, situaciones o secuencias que el paciente debio observar y retener). Actua como denominador en el calculo de ERS (Event Recall Score). Si se envia 0, ERS se fija en 0.

---

**`correct_events`** · `int >= 0` · Requerido

Cantidad de eventos narrativos que el paciente recordo o identifico correctamente. Debe ser menor o igual a `total_events`. Se usa para calcular ERS = `correct_events / total_events`.

---

**`comprehension_score`** · `int` · Valores: `0`, `1` o `2` · Requerido

Puntuacion cualitativa que representa el nivel de comprension narrativa del paciente sobre la historia o contexto de la sesion VR. La escala es:

| Valor | Significado |
|---|---|
| `0` | Sin comprension — el paciente no logro entender la narrativa |
| `1` | Comprension parcial — entendio parte del contexto |
| `2` | Comprension completa — entendio la narrativa en su totalidad |

Se normaliza internamente a SCS = `comprehension_score / 2`, obteniendo un valor entre 0.0 y 1.0.

---

**`response_times`** · `float[]` · Requerido

Lista de tiempos de respuesta individuales del paciente en segundos, uno por cada pregunta o interaccion evaluada durante la sesion. Puede contener cualquier cantidad de elementos (al menos uno es recomendable). Se calcula el promedio para obtener RTA (Response Time Average). Tiempos altos indican mayor latencia cognitiva; tiempos bajos indican respuesta rapida.

Ejemplo: `[2.1, 3.5, 1.8]` representa tres respuestas con tiempos de 2.1 s, 3.5 s y 1.8 s.

---

**`total_questions`** · `int >= 0` · Requerido

Cantidad total de preguntas realizadas al paciente durante o al finalizar la sesion VR. Actua como denominador en el calculo de ER (Error Rate). Si se envia 0, ER se fija en 0.

---

**`incorrect_answers`** · `int >= 0` · Requerido

Cantidad de preguntas que el paciente respondio incorrectamente. Debe ser menor o igual a `total_questions`. Se usa para calcular ER = `incorrect_answers / total_questions`. A mayor ER, peor el desempeno; ER alto penaliza directamente el SPS compuesto.

---

**`interaction_events`** · `int >= 0` · Requerido

Cantidad de interacciones que el paciente realizo efectivamente durante la sesion (por ejemplo: agarrar objetos, activar elementos, completar acciones en el entorno VR). Se compara contra `expected_interactions` para calcular ATS (Attention Score), que mide el nivel de participacion activa del paciente.

---

**`expected_interactions`** · `int >= 0` · Requerido

Cantidad de interacciones que se esperaba que el paciente realizara segun el diseno de la sesion VR. Actua como denominador en el calculo de ATS = `interaction_events / expected_interactions`. Si se envia 0, ATS se fija en 0. Un valor de ATS cercano a 1.0 indica que el paciente participo activamente; valores bajos sugieren desconexion o dificultad para interactuar con el entorno.

---

### Resumen de como se usan los parametros

| Parametro(s) | Metrica calculada | Formula |
|---|---|---|
| `correct_objects` / `total_objects` | ORS | `correct_objects / total_objects` |
| `correct_events` / `total_events` | ERS | `correct_events / total_events` |
| `comprehension_score` | SCS | `comprehension_score / 2` |
| `response_times` | RTA | `mean(response_times)` |
| `interaction_events` / `expected_interactions` | ATS | `interaction_events / expected_interactions` |
| `incorrect_answers` / `total_questions` | ER | `incorrect_answers / total_questions` |
| ORS, ERS, SCS, ER | SPS | `0.3·ORS + 0.3·ERS + 0.2·SCS + 0.2·(1−ER)` |

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
  "recommendation": "increase_difficulty",
  "context": {
    "baseline_sps": 0.62,
    "trend": "improving",
    "delta_sps": 0.19,
    "session_count": 7,
    "cold_start": false
  }
}
```

El campo `context` expone el razonamiento del motor de personalizacion: permite que Unity o el terapeuta vean por que se recomendo un ajuste especifico.

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

### Recomendacion de dificultad personalizada

El microservicio utiliza una arquitectura hibrida de dos capas:

**Capa 1 — Clasificacion SVM (stateless):** clasifica el rendimiento de la sesion actual en `low`, `medium` o `high`.

**Capa 2 — Motor de personalizacion clinica (stateful):** ajusta la recomendacion segun el historial longitudinal del paciente (ultimas 10 sesiones).

| Condicion evaluada | Recomendacion resultante |
|---|---|
| Sin historial previo (primera sesion) | Regla base por SPS |
| `delta_sps < -0.15` vs baseline personal | `decrease_difficulty` |
| Tendencia `declining` y base pedia `increase` | `maintain_difficulty` |
| Tendencia `improving` + SPS > 0.6 + base pedia `maintain` | `increase_difficulty` |
| Cualquier otro caso | Regla base: SPS < 0.4 → decrease, 0.4-0.7 → maintain, > 0.7 → increase |

### Mock Requests para Testing

Cuatro casos representativos que cubren los tres niveles de clasificacion. Todos corresponden a primera sesion del paciente (`cold_start: true`), por lo que la recomendacion usa la regla base de SPS.

---

**Caso 1 — Rendimiento alto** (`SPS ≈ 0.89` → `high` → `increase_difficulty`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 101,
    "total_objects": 10,
    "correct_objects": 9,
    "total_events": 10,
    "correct_events": 8,
    "comprehension_score": 2,
    "response_times": [1.5, 2.0, 1.8, 1.6],
    "total_questions": 10,
    "incorrect_answers": 1,
    "interaction_events": 9,
    "expected_interactions": 10
  }'
```

Respuesta esperada:
```json
{
  "metrics": { "ors": 0.9, "ers": 0.8, "scs": 1.0, "rta": 1.725, "ats": 0.9, "er": 0.1, "sps": 0.89 },
  "prediction": "high",
  "recommendation": "increase_difficulty",
  "context": { "baseline_sps": 0.89, "trend": "cold_start", "delta_sps": 0.0, "session_count": 0, "cold_start": true }
}
```

---

**Caso 2 — Rendimiento medio** (`SPS ≈ 0.57` → `medium` → `maintain_difficulty`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 102,
    "total_objects": 10,
    "correct_objects": 6,
    "total_events": 10,
    "correct_events": 5,
    "comprehension_score": 1,
    "response_times": [3.5, 4.2, 3.8, 4.5],
    "total_questions": 10,
    "incorrect_answers": 3,
    "interaction_events": 6,
    "expected_interactions": 10
  }'
```

Respuesta esperada:
```json
{
  "metrics": { "ors": 0.6, "ers": 0.5, "scs": 0.5, "rta": 4.0, "ats": 0.6, "er": 0.3, "sps": 0.57 },
  "prediction": "medium",
  "recommendation": "maintain_difficulty",
  "context": { "baseline_sps": 0.57, "trend": "cold_start", "delta_sps": 0.0, "session_count": 0, "cold_start": true }
}
```

---

**Caso 3 — Rendimiento bajo** (`SPS ≈ 0.16` → `low` → `decrease_difficulty`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 103,
    "total_objects": 10,
    "correct_objects": 2,
    "total_events": 10,
    "correct_events": 2,
    "comprehension_score": 0,
    "response_times": [6.0, 7.5, 8.0, 6.8],
    "total_questions": 10,
    "incorrect_answers": 8,
    "interaction_events": 2,
    "expected_interactions": 10
  }'
```

Respuesta esperada:
```json
{
  "metrics": { "ors": 0.2, "ers": 0.2, "scs": 0.0, "rta": 7.075, "ats": 0.2, "er": 0.8, "sps": 0.16 },
  "prediction": "low",
  "recommendation": "decrease_difficulty",
  "context": { "baseline_sps": 0.16, "trend": "cold_start", "delta_sps": 0.0, "session_count": 0, "cold_start": true }
}
```

---

**Caso 4 — Limite medio-alto** (`SPS ≈ 0.78` → `high` → `increase_difficulty`)

Util para verificar que el modelo clasifica correctamente en el borde entre `medium` y `high`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 104,
    "total_objects": 10,
    "correct_objects": 7,
    "total_events": 10,
    "correct_events": 7,
    "comprehension_score": 2,
    "response_times": [2.5, 3.0, 2.8],
    "total_questions": 10,
    "incorrect_answers": 2,
    "interaction_events": 7,
    "expected_interactions": 10
  }'
```

Respuesta esperada:
```json
{
  "metrics": { "ors": 0.7, "ers": 0.7, "scs": 1.0, "rta": 2.767, "ats": 0.7, "er": 0.2, "sps": 0.78 },
  "prediction": "high",
  "recommendation": "increase_difficulty",
  "context": { "baseline_sps": 0.78, "trend": "cold_start", "delta_sps": 0.0, "session_count": 0, "cold_start": true }
}
```

> Para observar la personalizacion en accion, enviar multiples requests con el mismo `patient_id` en orden. A partir del segundo request, `cold_start` sera `false` y el motor ajustara la recomendacion segun el historial acumulado.

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
public class PatientContext
{
    public float baseline_sps;
    public string trend;
    public float delta_sps;
    public int session_count;
    public bool cold_start;
}

[System.Serializable]
public class PredictionResponse
{
    public SessionMetrics metrics;
    public string prediction;
    public string recommendation;
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
            Debug.Log($"Prediction: {response.prediction}");
            Debug.Log($"Recommendation: {response.recommendation}");
            Debug.Log($"SPS: {response.metrics.sps}");
            Debug.Log($"Trend: {response.context.trend}");
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
│   ├── session_metrics.py           # RawSessionData, SessionMetrics, CognitiveLevel
│   ├── patient_context.py           # HistoricalSession, PatientContext, TrendType
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
│   │   ├── onnx_classifier.py       # Wrapper SVM ONNX + scaler
│   │   └── personalization_engine.py# Motor de reglas clinicas de personalizacion
│   └── persistence/
│       ├── postgres_pool.py         # Pool asyncpg + DDL
│       └── session_repository.py    # insert_session, get_patient_history
│
├── interfaces/                      # Puntos de entrada HTTP
│   └── http/
│       ├── api.py                   # FastAPI app + lifespan + endpoint /predict
│       └── schemas.py               # Pydantic DTOs de entrada y salida
│
└── misc/                            # Artefactos de ML y herramientas de entrenamiento
    ├── train.py                     # Script de entrenamiento y exportacion a ONNX
    ├── model.onnx                   # Modelo SVM exportado (generado por train.py)
    ├── scaler.joblib                # StandardScaler ajustado (generado por train.py)
    └── dataset/
        └── synthetic_vr_dataset.csv # Dataset de entrenamiento
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

El microservicio persiste cada sesion en el esquema `schema_telemetria` de PostgreSQL (TimescaleDB). La tabla `metricas_sesion` se crea automaticamente al iniciar el servidor e incluye las 7 metricas cognitivas, la prediccion, la recomendacion, y el contexto de personalizacion usado (`baseline_sps`, `trend`, `delta_sps`, `session_count`). Ver `infrastructure/persistence/postgres_pool.py` para el DDL completo.
