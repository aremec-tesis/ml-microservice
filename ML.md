# Machine Learning en el Microservicio de Rendimiento Cognitivo

## Tecnica utilizada

Este microservicio implementa un **Support Vector Machine (SVM)** con kernel RBF (Radial Basis Function), un algoritmo de clasificacion supervisada ampliamente validado en la literatura cientifica. El SVM fue seleccionado por su robustez ante espacios de features de baja-media dimensionalidad, su capacidad de generalizacion con datasets pequenos, y su interpretabilidad frente a alternativas de deep learning.

A diferencia de versiones anteriores, el modelo ahora es **stateful**: consume tanto las 6 metricas de la sesion actual como **9 features agregadas del historial longitudinal del paciente**, y predice directamente la **recomendacion de dificultad** (decrease / maintain / increase). El motor de personalizacion clinica externo desaparece: su logica fue absorbida por el modelo, que la aprendio del dataset etiquetado con esas mismas reglas.

## Pipeline de inferencia

```
Sesion VR (Unity)
       |
       v
Validacion Pydantic              <-  interfaces/http/schemas.py (SessionInput)
       |
       v
Lectura de historial del paciente <-  app/queries/get_patient_history.py
                                     -> infrastructure/persistence/session_repository.py
       |
       v
Calculo de metricas cognitivas   <-  domain/session_metrics.py (SessionMetrics.from_raw)
  [ORS, ERS, SCS, RTA, ER, SPS]
       |
       v
Contexto longitudinal             <-  domain/patient_context.py (PatientContext.from_history)
  [baseline_sps, slope_sps, delta_sps, mean_ors, mean_ers, mean_er, mean_rta, std_sps, session_count]
       |
       v
Feature vector (15 features)      <-  domain/patient_context.feature_vector
       |
       v
Normalizacion (StandardScaler)    <-  scaler.joblib
       |
       v
Inferencia SVM stateful (ONNX)    <-  infrastructure/ml/onnx_classifier.py
       |                               +-> Recomendacion: decrease / maintain / increase
       |                               +-> Probabilidades por clase (trazabilidad clinica)
       v
Cognitive level derivado del SPS  <-  domain/session_metrics.cognitive_level_from_sps
  (informacion clinica para el terapeuta, no afecta al ML)
       |
       v
Persistencia de sesion            <-  infrastructure/persistence/session_repository.py
  (raw + metricas + 15 features usadas + recomendacion + probabilidades + cognitive_level)
       |
       v
Respuesta JSON                    ->  POST /predict
  (metrics + cognitive_level + recommendation + probabilities + context)
```

Toda la orquestacion vive en `app/commands/predict_session.py` (comando CQRS) para que la capa HTTP sea delgada y el flujo sea trivialmente testeable sin FastAPI.

El modelo se entrena **una sola vez** con el script `misc/train.py`, se exporta al formato estandar ONNX, y en produccion el microservicio lo carga en memoria para inferencia eficiente sin dependencia de scikit-learn.

## Enfoque stateful con trazabilidad clinica

El modelo recibe las 15 features y produce directamente la recomendacion de dificultad. Para mantener defensibilidad clinica sin un motor de reglas externo, el sistema:

1. **Persiste las 15 features de entrada** en cada inferencia (`schema_telemetria.metricas_sesion`), por lo que cualquier prediccion puede reconstruirse a partir de la base de datos.
2. **Persiste las 3 probabilidades por clase** (`prob_decrease`, `prob_maintain`, `prob_increase`), permitiendo al terapeuta entender la confianza del modelo en cada decision (ej. "el modelo dio 0.78 a maintain vs 0.15 a increase").
3. **El dataset sintetico fue etiquetado con reglas clinicas defendibles** (las mismas reglas que tenia el motor de personalizacion previo), por lo que el modelo aprendio a replicar esas reglas pero usando el espacio completo de features. Esto permite migrar progresivamente hacia datos clinicos reales sin reescribir la arquitectura.

## Las 15 features del modelo

| # | Feature | Origen | Descripcion |
|---|---------|--------|-------------|
| 1 | ORS | Sesion actual | Object Recall Score con nueva formula clave/secundario |
| 2 | ERS | Sesion actual | Event Recall Score |
| 3 | SCS | Sesion actual | Semantic Comprehension Score |
| 4 | RTA | Sesion actual | Response Time Average (segundos) |
| 5 | ER | Sesion actual | Error Rate |
| 6 | SPS | Sesion actual | Session Performance Score (compuesto) |
| 7 | baseline_sps | Historial | Media movil ponderada del SPS sobre las ultimas 10 sesiones |
| 8 | slope_sps | Historial | Pendiente lineal del SPS (positivo = mejora, negativo = declive) |
| 9 | delta_sps | Historial | SPS_actual - baseline_sps |
| 10 | mean_ors | Historial | Promedio del ORS historico |
| 11 | mean_ers | Historial | Promedio del ERS historico |
| 12 | mean_er | Historial | Promedio del ER historico |
| 13 | mean_rta | Historial | Promedio del RTA historico |
| 14 | std_sps | Historial | Desviacion estandar del SPS historico (volatilidad) |
| 15 | session_count | Historial | Numero de sesiones previas usadas (0 en cold start) |

En **cold start** (paciente sin historial), las features de historial se neutralizan: `baseline_sps = SPS_actual`, `slope_sps = 0`, `delta_sps = 0`, `std_sps = 0`, `mean_* = valores actuales`, `session_count = 0`. El modelo aprendio durante el entrenamiento que `session_count = 0` indica cold start y se comporta de forma conservadora en esos casos.

## Nueva formula de ORS

```
ORS = ((correct_key * 2) + (correct_secondary * 1) - (incorrect * 1))
      ────────────────────────────────────────────────────────────────
                  (total_key * 2) + (total_secondary * 1)
```

A diferencia de la version previa (`correct / total`), esta formula:
- **Pondera doble los objetos clave** sobre los secundarios, reflejando su importancia clinica/narrativa.
- **Penaliza objetos incorrectamente identificados** (falsos positivos) restandolos del numerador.
- **Puede ser negativa** cuando los errores superan los aciertos ponderados — el modelo y el scaler manejan rango ampliado sin problemas.

## Sobre el dataset sintetico

El dataset (`misc/dataset/synthetic_vr_dataset.csv`) es **longitudinal**: contiene multiples sesiones por paciente para que el modelo pueda aprender la influencia del historial sobre la recomendacion. Generador en `misc/generate_dataset.py`.

- **500 pacientes**, cada uno con 5-20 sesiones (~6000 filas totales).
- Por paciente, un **fenotipo cognitivo** (improving / stable / declining) define la trayectoria latente.
- Por sesion, las 6 metricas se muestrean con coherencia respecto a la habilidad cognitiva del paciente en ese momento (drift + ruido).
- Las 9 features de historial se calculan para cada sesion usando solo las sesiones previas reales del mismo paciente (no hay leakage temporal).
- El target `Target_Recommendation` se etiqueta con reglas clinicas defendibles (umbrales de SPS + slope + delta), las mismas reglas del motor de personalizacion previo.

El split train/test en `train.py` es **estratificado por paciente** para evitar que sesiones del mismo paciente aparezcan a ambos lados del split (data leakage).

## Punto de distincion del proyecto

El hecho de que no exista un dataset previo para este problema especifico no es una limitacion — es evidencia directa de la **novedad del aporte**. El proyecto define un nuevo conjunto de metricas cognitivas derivadas de sesiones VR, propone un modelo stateful que personaliza por paciente sin necesidad de un motor de reglas externo, persiste cada decision con trazabilidad completa, y genera los datos necesarios para validarlo. Esto representa una contribucion metodologica independiente al dominio de la neurorrehabilitacion asistida por tecnologia.
