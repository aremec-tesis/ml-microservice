---
task: Explain ML technique used in VR microservice
slug: 20260416-000001_explain-ml-technique-vr-microservice
effort: standard
phase: execute
progress: 8/8
mode: interactive
started: 2026-04-16T00:00:01Z
updated: 2026-04-16T00:01:00Z
---

## Context

Mauricio necesita entender (sin ser experto en ML) qué técnica de machine learning usa el microservicio, si es entrenamiento o fine-tuning, y la situación del dataset sintético.

## Criteria

- [x] ISC-1: Técnica SVM identificada y nombrada correctamente
- [x] ISC-2: Proceso de entrenamiento supervisado explicado
- [x] ISC-3: Rol de StandardScaler explicado
- [x] ISC-4: Propósito del export ONNX explicado
- [x] ISC-5: Vector de features [ORS,ERS,SCS,RTA,ATS,ER,SPS] explicado
- [x] ISC-6: Labels Target_Class (low/medium/high) explicados
- [x] ISC-7: Situación del dataset sintético explicada
- [x] ISC-8: Diferencia training vs fine-tuning vs transfer learning aclarada

## Verification

Todos los criterios cubiertos en la respuesta al usuario, verificado por lectura de train.py, model_handler.py, metrics.py, main.py y dataset.
