"""
Microservicio de prediccion de rendimiento cognitivo.

Recibe metricas de desempeno desde una aplicacion Unity de estimulacion
cognitiva y predice el nivel de rendimiento del paciente (bajo/medio/alto)
usando un modelo SVM.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- Rutas de archivos del modelo ---

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "modelo_svm.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

ETIQUETAS = {0: "bajo", 1: "medio", 2: "alto"}

# --- Variables globales para modelo y scaler ---

modelo: SVC | None = None
scaler: StandardScaler | None = None


# =====================================================================
# Modelos Pydantic
# =====================================================================

class MetricasPaciente(BaseModel):
    """Metricas de desempeno recibidas desde la aplicacion Unity."""

    tiempo_reaccion: float = Field(
        ..., gt=0, description="Tiempo de reaccion en milisegundos"
    )
    aciertos: int = Field(
        ..., ge=0, description="Numero de respuestas correctas"
    )
    errores: int = Field(
        ..., ge=0, description="Numero de respuestas incorrectas"
    )
    tiempo_total: float = Field(
        ..., gt=0, description="Tiempo total de la sesion en segundos"
    )


class PrediccionResponse(BaseModel):
    """Respuesta con la prediccion de rendimiento cognitivo."""

    rendimiento_cognitivo: str


# =====================================================================
# Entrenamiento del modelo
# =====================================================================

def entrenar_modelo() -> tuple[SVC, StandardScaler]:
    """Entrena un modelo SVM con datos simulados y lo guarda en disco."""

    rng = np.random.default_rng(42)
    n_por_clase = 60

    # Datos simulados por clase:
    # [tiempo_reaccion, aciertos, errores, tiempo_total]

    # Bajo: reaccion lenta, pocos aciertos, muchos errores, sesion larga
    bajo = np.column_stack([
        rng.normal(2000, 400, n_por_clase),   # tiempo_reaccion ~2000ms
        rng.normal(3, 1.5, n_por_clase),      # aciertos ~3
        rng.normal(6, 1.5, n_por_clase),      # errores ~6
        rng.normal(450, 60, n_por_clase),      # tiempo_total ~450s
    ])

    # Medio: valores intermedios
    medio = np.column_stack([
        rng.normal(1200, 300, n_por_clase),   # tiempo_reaccion ~1200ms
        rng.normal(6, 1.5, n_por_clase),      # aciertos ~6
        rng.normal(3, 1.5, n_por_clase),      # errores ~3
        rng.normal(300, 50, n_por_clase),      # tiempo_total ~300s
    ])

    # Alto: reaccion rapida, muchos aciertos, pocos errores, sesion corta
    alto = np.column_stack([
        rng.normal(600, 200, n_por_clase),    # tiempo_reaccion ~600ms
        rng.normal(9, 1, n_por_clase),        # aciertos ~9
        rng.normal(1, 0.8, n_por_clase),      # errores ~1
        rng.normal(180, 40, n_por_clase),      # tiempo_total ~180s
    ])

    X = np.vstack([bajo, medio, alto])
    y = np.array([0] * n_por_clase + [1] * n_por_clase + [2] * n_por_clase)

    # Escalar las features
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # Entrenar SVM
    svm = SVC(kernel="rbf", C=1.0, random_state=42)
    svm.fit(X_scaled, y)

    # Guardar modelo y scaler
    joblib.dump(svm, MODELO_PATH)
    joblib.dump(sc, SCALER_PATH)

    print(f"Modelo entrenado y guardado en {MODELO_PATH}")
    print(f"Scaler guardado en {SCALER_PATH}")

    return svm, sc


# =====================================================================
# Ciclo de vida de la aplicacion
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga (o entrena) el modelo al iniciar la aplicacion."""

    global modelo, scaler

    if not MODELO_PATH.exists() or not SCALER_PATH.exists():
        print("Modelo no encontrado. Entrenando modelo con datos simulados...")
        modelo, scaler = entrenar_modelo()
    else:
        modelo = joblib.load(MODELO_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Modelo cargado correctamente.")

    yield


# =====================================================================
# Aplicacion FastAPI
# =====================================================================

app = FastAPI(
    title="API de Rendimiento Cognitivo",
    description=(
        "Microservicio que predice el nivel de rendimiento cognitivo "
        "de pacientes con Alzheimer a partir de metricas de desempeno "
        "en una aplicacion de realidad virtual."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PrediccionResponse)
async def predecir(metricas: MetricasPaciente):
    """Predice el rendimiento cognitivo a partir de las metricas del paciente."""

    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="El modelo no esta disponible.")

    try:
        datos = np.array([[
            metricas.tiempo_reaccion,
            metricas.aciertos,
            metricas.errores,
            metricas.tiempo_total,
        ]])

        datos_escalados = scaler.transform(datos)
        prediccion = modelo.predict(datos_escalados)[0]
        etiqueta = ETIQUETAS[int(prediccion)]

        return PrediccionResponse(rendimiento_cognitivo=etiqueta)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la prediccion: {e}")


# =====================================================================
# Ejecucion directa
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
