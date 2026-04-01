# API de Rendimiento Cognitivo

Microservicio en Python con FastAPI que predice el nivel de rendimiento cognitivo de pacientes con Alzheimer a partir de metricas de desempeno capturadas en una aplicacion de realidad virtual desarrollada en Unity.

Utiliza un modelo de clasificacion SVM (Support Vector Machine) entrenado con scikit-learn que clasifica el rendimiento en tres niveles: **bajo**, **medio** y **alto**.

## Requisitos

- Python 3.10 o superior

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

## Ejecucion

```bash
python main.py
```

El servidor se inicia en `http://localhost:8000`. En la primera ejecucion, el modelo SVM se entrena automaticamente con datos simulados y se guarda en disco (`modelo_svm.pkl` y `scaler.pkl`).

## Uso

### Endpoint

**POST** `/predict`

### Request

```json
{
  "tiempo_reaccion": 1200,
  "aciertos": 8,
  "errores": 2,
  "tiempo_total": 300
}
```

| Campo | Tipo | Validacion | Descripcion |
|---|---|---|---|
| `tiempo_reaccion` | float | > 0 | Tiempo de reaccion en milisegundos |
| `aciertos` | int | >= 0 | Numero de respuestas correctas |
| `errores` | int | >= 0 | Numero de respuestas incorrectas |
| `tiempo_total` | float | > 0 | Duracion total de la sesion en segundos |

### Response

```json
{
  "rendimiento_cognitivo": "medio"
}
```

Valores posibles: `"bajo"`, `"medio"`, `"alto"`.

### Ejemplo con curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tiempo_reaccion": 1200, "aciertos": 8, "errores": 2, "tiempo_total": 300}'
```

### Ejemplo en Unity (C#)

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System.Collections;

[System.Serializable]
public class MetricasPaciente
{
    public float tiempo_reaccion;
    public int aciertos;
    public int errores;
    public float tiempo_total;
}

[System.Serializable]
public class PrediccionResponse
{
    public string rendimiento_cognitivo;
}

public class CognitiveAPI : MonoBehaviour
{
    private string apiUrl = "http://localhost:8000/predict";

    public IEnumerator EnviarPrediccion(float tiempoReaccion, int aciertos, int errores, float tiempoTotal)
    {
        var metricas = new MetricasPaciente
        {
            tiempo_reaccion = tiempoReaccion,
            aciertos = aciertos,
            errores = errores,
            tiempo_total = tiempoTotal
        };

        string json = JsonUtility.ToJson(metricas);
        byte[] body = Encoding.UTF8.GetBytes(json);

        using var request = new UnityWebRequest(apiUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(body);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            var respuesta = JsonUtility.FromJson<PrediccionResponse>(request.downloadHandler.text);
            Debug.Log("Rendimiento cognitivo: " + respuesta.rendimiento_cognitivo);
        }
        else
        {
            Debug.LogError("Error: " + request.error);
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
ml-microservice/
├── main.py              # Microservicio (entrenamiento + API)
├── requirements.txt     # Dependencias
├── modelo_svm.pkl       # Modelo SVM entrenado (generado automaticamente)
├── scaler.pkl           # Scaler de normalizacion (generado automaticamente)
└── README.md
```

## Tecnologias

- **FastAPI** - Framework web
- **scikit-learn** - Modelo SVM de clasificacion
- **joblib** - Serializacion del modelo
- **pydantic** - Validacion de datos
- **uvicorn** - Servidor ASGI
