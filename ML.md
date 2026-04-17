# Machine Learning en el Microservicio de Rendimiento Cognitivo

## Técnica utilizada

Este microservicio implementa un **Support Vector Machine (SVM)** con kernel RBF (Radial Basis Function), un algoritmo de clasificación supervisada ampliamente validado en la literatura científica. El SVM fue seleccionado por su robustez ante espacios de features de baja dimensionalidad, su capacidad de generalización con datasets pequeños, y su interpretabilidad frente a alternativas de deep learning.

El modelo clasifica cada sesión VR en uno de tres niveles de rendimiento cognitivo: `low`, `medium` o `high`, a partir de 7 métricas calculadas en tiempo real desde los datos de la sesión.

## Pipeline de inferencia

```
Sesión VR (Unity)
       │
       ▼
Validación Pydantic              ←  interfaces/http/schemas.py (SessionInput)
       │
       ▼
Lectura de historial del paciente ←  app/queries/get_patient_history.py
                                     → infrastructure/persistence/session_repository.py
       │
       ▼
Cálculo de métricas cognitivas   ←  domain/session_metrics.py (SessionMetrics.from_raw)
  [ORS, ERS, SCS, RTA, ATS, ER, SPS]
       │
       ▼
Contexto longitudinal             ←  domain/patient_context.py (PatientContext.from_history)
  (baseline + tendencia + delta)
       │
       ▼
Normalización (StandardScaler)    ←  scaler.joblib
       │
       ▼
Inferencia SVM (ONNX Runtime)     ←  infrastructure/ml/onnx_classifier.py
       │                               └─→  Clasificación: low / medium / high
       ▼
Motor de personalización clínica  ←  infrastructure/ml/personalization_engine.py
                                     reglas interpretables + contexto del paciente
       │
       ▼
Recomendación personalizada       →  increase / maintain / decrease
       │
       ▼
Persistencia de sesión            ←  infrastructure/persistence/session_repository.py
  (incluye contexto usado)            insert con baseline_sps, trend, delta_sps, session_count
       │
       ▼
Respuesta JSON                    →  POST /predict
  (metrics + prediction + recommendation + context)
```

Toda la orquestación vive en `app/commands/predict_session.py` (comando CQRS) para que la capa HTTP sea delgada y el flujo sea trivialmente testeable sin FastAPI.

El modelo se entrena **una sola vez** con el script `train.py`, se exporta al formato estándar ONNX, y en producción el microservicio lo carga en memoria para inferencia eficiente sin dependencia de scikit-learn.

## Enfoque híbrido — ML clasificatorio + motor de personalización clínica

La propuesta central del proyecto es el **ajuste de dificultad totalmente personalizado** para cada paciente. Para lograrlo de forma defendible y clínicamente interpretable, el microservicio adopta una arquitectura **híbrida de dos capas** dentro del endpoint `/predict`:

**Capa 1 — Clasificación ML (SVM objetivo)**
El SVM clasifica el rendimiento cognitivo de la sesión actual (low / medium / high) usando exclusivamente las 7 métricas de esa sesión. Esta capa es **estateless y objetiva**: dos pacientes con el mismo desempeño en una sesión reciben la misma clasificación.

**Capa 2 — Motor de personalización clínica (stateful)**
Sobre la clasificación del SVM, un motor de reglas interpretables combina la predicción con el **historial longitudinal del paciente** recuperado de la base de datos TimescaleDB para producir una recomendación de dificultad **personalizada a la trayectoria individual** del paciente. El motor considera:

- **Baseline del paciente**: media móvil ponderada del SPS sobre las últimas N sesiones
- **Tendencia**: pendiente de SPS sobre la ventana de historial → `improving` / `stable` / `declining`
- **Delta relativo**: diferencia entre el SPS actual y el baseline personal del paciente
- **Número de sesiones previas**: modula cuánta influencia tiene el historial (cold-start)

Esta separación garantiza que:

1. **El modelo ML permanece puro, reentrenable y auditable** — su rol es clasificar, no personalizar.
2. **La personalización es transparente y defendible clínicamente** — cada ajuste de dificultad puede trazarse a reglas interpretables basadas en literatura de neurorrehabilitación.
3. **El sistema evoluciona de forma segura hacia datos reales** — cuando existan datos clínicos longitudinales, la capa de personalización puede migrar progresivamente hacia features del propio modelo ML sin reescribir el sistema.

En contextos clínicos, los sistemas híbridos ML + reglas interpretables son ampliamente preferidos frente a modelos caja-negra, precisamente porque un terapeuta debe poder entender **por qué** el sistema recomendó aumentar o disminuir la dificultad. Este enfoque híbrido no es una limitación del proyecto: es una decisión de diseño alineada con buenas prácticas de IA aplicada a la salud.

### Reglas concretas del motor de personalización

| Condición | Ajuste aplicado |
|---|---|
| Paciente sin historial (cold start) | Regla base SPS sin personalización |
| `delta_sps < -0.15` respecto al baseline | `decrease_difficulty` (caída significativa) |
| Tendencia `declining` y regla base pedía `increase` | Se fuerza `maintain_difficulty` (no presionar) |
| Tendencia `improving` + regla base `maintain` + `SPS > 0.6` | Se fuerza `increase_difficulty` (premiar progreso) |
| Cualquier otro caso | Regla base por SPS |

Los umbrales están definidos como constantes documentadas en `infrastructure/ml/personalization_engine.py` y pueden calibrarse sin modificar la lógica.

### Contenido de la respuesta del endpoint

La respuesta incluye el contexto clínico usado, permitiendo que Unity o el terapeuta visualicen el razonamiento detrás de cada recomendación:

```json
{
  "metrics": { "ors": 0.70, "ers": 0.80, "scs": 1.0, "rta": 2.46, "ats": 0.80, "er": 0.20, "sps": 0.81 },
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

Esto convierte al sistema en una herramienta de apoyo clínico transparente: cada decisión queda trazada y auditada en la base de datos (`schema_telemetria.metricas_sesion` guarda el contexto usado en cada inferencia).

## Features del modelo

| Feature | Descripción |
|---|---|
| ORS | Ratio de objetos reconocidos correctamente |
| ERS | Ratio de eventos reconocidos correctamente |
| SCS | Puntuación de comprensión semántica |
| RTA | Tiempo de respuesta promedio (segundos) |
| ATS | Ratio de interacciones realizadas vs esperadas |
| ER | Tasa de error en respuestas |
| SPS | Puntuación compuesta de desempeño de sesión |

## Sobre el dataset sintético

El proyecto no cuenta con un dataset público de referencia porque **no existe uno**. El enfoque del proyecto es novedoso: la estimulación cognitiva guiada por realidad virtual con métricas de sesión como proxy de rendimiento cognitivo es una propuesta original de esta tesis, sin antecedentes directos en literatura con exactamente estas variables.

Ante esta situación, el equipo tomó una decisión técnicamente fundamentada: **construir un dataset sintético** (`dataset/synthetic_vr_dataset.csv`) con correlaciones realistas entre las métricas de sesión VR y el nivel de rendimiento cognitivo, apoyado en escalas clínicas establecidas como el MMSE (Mini-Mental State Examination).

Este enfoque es una práctica reconocida en investigación cuando el fenómeno de estudio es emergente y no dispone de datos históricos. Permite:

- Validar la arquitectura del pipeline completo (desde Unity hasta el modelo)
- Demostrar el funcionamiento end-to-end del sistema
- Sentar la base para reemplazar los datos sintéticos por datos reales en fases clínicas posteriores del proyecto

La columna `Target_Class` del dataset fue construida de forma coherente con los valores de las métricas, asegurando que el modelo aprenda patrones que reflejan el comportamiento clínico esperado.

## Punto de distinción del proyecto

El hecho de que no exista un dataset previo para este problema específico no es una limitación — es evidencia directa de la **novedad del aporte**. El proyecto define un nuevo conjunto de métricas cognitivas derivadas de sesiones VR, propone un pipeline de predicción automatizado, y genera los datos necesarios para validarlo. Esto representa una contribución metodológica independiente al dominio de la neurorrehabilitación asistida por tecnología.
