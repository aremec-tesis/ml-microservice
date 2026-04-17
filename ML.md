# Machine Learning en el Microservicio de Rendimiento Cognitivo

## Técnica utilizada

Este microservicio implementa un **Support Vector Machine (SVM)** con kernel RBF (Radial Basis Function), un algoritmo de clasificación supervisada ampliamente validado en la literatura científica. El SVM fue seleccionado por su robustez ante espacios de features de baja dimensionalidad, su capacidad de generalización con datasets pequeños, y su interpretabilidad frente a alternativas de deep learning.

El modelo clasifica cada sesión VR en uno de tres niveles de rendimiento cognitivo: `low`, `medium` o `high`, a partir de 7 métricas calculadas en tiempo real desde los datos de la sesión.

## Pipeline de inferencia

```
Sesión VR (Unity)
       │
       ▼
Cálculo de métricas cognitivas  ←  metrics.py
  [ORS, ERS, SCS, RTA, ATS, ER, SPS]
       │
       ▼
Normalización (StandardScaler)  ←  scaler.joblib
       │
       ▼
Inferencia SVM (ONNX Runtime)   ←  model.onnx
       │
       ▼
Clasificación + Recomendación   →  POST /predict
```

El modelo se entrena **una sola vez** con el script `train.py`, se exporta al formato estándar ONNX, y en producción el microservicio lo carga en memoria para inferencia eficiente sin dependencia de scikit-learn.

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
