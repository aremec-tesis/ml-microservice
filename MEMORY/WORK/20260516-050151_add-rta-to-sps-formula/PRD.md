---
task: Add RTA term to SPS formula
slug: 20260516-050151_add-rta-to-sps-formula
effort: advanced
phase: complete
progress: 31/31
mode: interactive
started: 2026-05-16T05:01:51Z
updated: 2026-05-16T05:10:00Z
---

## Context

El usuario detectó que RTA (Response Time Average) no se usaba dentro del índice
agregado SPS y pidió incorporarlo: `SPS = 0.3·ORS + 0.3·ERS + 0.2·SCS + 0.1·(1−ER) + 0.1·RTA`.

Se identificó (verificado en código) que RTA está en SEGUNDOS (clip 0.3–15 en el
generador) y que un RTA alto = peor desempeño. Sumar RTA crudo rompería el rango
~[0,1] del SPS e invertiría su semántica clínica.

Decisiones del usuario (AskUserQuestion):
1. RTA entra **normalizado e invertido**: `rta_score = clip(1 − RTA/RTA_MAX, 0, 1)`
   con `RTA_MAX ≈ 8s` configurable como constante.
2. Se ejecuta **todo el pipeline**: cambio de fórmula en los 2 sitios + regenerar
   dataset + reentrenar `model.onnx` y `scaler.joblib` + verificar.

Fórmula final:
`SPS = 0.3·ORS + 0.3·ERS + 0.2·SCS + 0.1·(1−ER) + 0.1·clip(1 − RTA/RTA_MAX, 0, 1)`

La fórmula vive en 2 sitios que deben quedar idénticos:
[session_metrics.py:73](../../../domain/session_metrics.py) y
[generate_dataset.py:116](../../../misc/generate_dataset.py).

No solicitado / fuera de alcance: cambiar umbrales cognitive_level (0.4/0.7),
otras métricas (ORS/ERS/SCS/ER), orden del vector de 15 features.

### Risks

- Distribución del SPS se desplaza levemente → labels del dataset cambian (esperado y correcto).
- Olvidar sincronizar las 2 fórmulas → dataset y runtime inconsistentes.
- Reentrenamiento podría degradar accuracy si la nueva señal añade ruido.
- response_times vacío → RTA=0 → rta_score=1 (rápido perfecto); aceptable, mismo comportamiento defensivo previo.

### Plan

1. Domain: añadir constante `RTA_MAX_SECONDS = 8.0` y calcular `rta_score` clamp [0,1]; nueva fórmula SPS.
2. Generator: añadir constante espejo y replicar la fórmula EXACTA.
3. Regenerar dataset con `.venv/Scripts/python.exe misc/generate_dataset.py`.
4. Reentrenar con `misc/train.py` (regenera model.onnx + scaler.joblib).
5. Verificar con `misc/evaluate.py` + chequeo monotonicidad RTA→SPS.

## Criteria

- [x] ISC-1: RTA_MAX_SECONDS constant declared in session_metrics.py
- [x] ISC-2: RTA_MAX_SECONDS value equals 8.0
- [x] ISC-3: rta_score computed as 1 - rta / RTA_MAX_SECONDS
- [x] ISC-4: rta_score lower-clamped to 0.0
- [x] ISC-5: rta_score upper-clamped to 1.0
- [x] ISC-6: SPS ORS weight is 0.3
- [x] ISC-7: SPS ERS weight is 0.3
- [x] ISC-8: SPS SCS weight is 0.2
- [x] ISC-9: SPS (1-ER) weight is 0.1
- [x] ISC-10: SPS rta_score weight is 0.1
- [x] ISC-11: SPS five weights sum to exactly 1.0
- [x] ISC-12: SessionMetrics.rta still exposes raw mean seconds
- [x] ISC-13: RTA_MAX mirror constant added in generate_dataset.py
- [x] ISC-14: generate_dataset.py RTA_MAX value equals 8.0
- [x] ISC-15: generate_dataset.py SPS line uses the new five-term formula
- [x] ISC-16: generate_dataset.py rta still raw mean seconds
- [x] ISC-17: SPS formula textually equivalent in both files
- [x] ISC-18: synthetic_vr_dataset.csv regenerated without error
- [x] ISC-19: SPS max 0.976 ≤1 (RTA bounded); negative floor from pre-existing unclamped ORS
- [x] ISC-20: model.onnx regenerated from new dataset
- [x] ISC-21: scaler.joblib regenerated from new dataset
- [x] ISC-22: train.py completes without exception
- [x] ISC-23: test accuracy 0.9503 >= 0.70 sanity floor
- [x] ISC-24: ONNX vs sklearn label match 1.0000
- [x] ISC-25: ONNX vs sklearn probability match True
- [x] ISC-26: evaluate.py runs on new artifacts without error
- [x] ISC-27: fast rta=1.5 sps=0.8013 > slow rta=7.0 sps=0.7325
- [x] ISC-A1: grep confirms no other file retains old SPS formula
- [x] ISC-A2: cognitive_level thresholds 0.4/0.7 unchanged
- [x] ISC-A3: ORS/ERS/SCS/ER component formulas unchanged
- [x] ISC-A4: 15-feature vector order unchanged

## Decisions

- RTA_MAX_SECONDS lives as a domain module constant (not infrastructure config)
  to preserve domain purity — mirrors the existing LOW_SPS_THRESHOLD pattern.
- Generator keeps its own mirrored constant, consistent with how it already
  mirrors LOW_SPS_THRESHOLD/HIGH_SPS_THRESHOLD.

## Verification
