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

### Descripcion de parametros

---

**`patient_id`** · `int` · Requerido

Identificador numerico unico del paciente dentro del sistema. Se usa para recuperar el historial de sesiones anteriores (hasta 10 sesiones) que el modelo ML consume como features adicionales para producir una recomendacion personalizada. No se valida su existencia previa — si es la primera sesion del paciente, el sistema opera en modo cold start y las features de historial se neutralizan.

---

**`correct_key_objects`** · `int >= 0` · Requerido

Cantidad de objetos **clave** (alta importancia clinica/narrativa) que el paciente identifico o recordo correctamente. Pondera doble en el calculo de ORS. Debe ser menor o igual a `total_key_objects`.

---

**`correct_secondary_objects`** · `int >= 0` · Requerido

Cantidad de objetos **secundarios** (importancia menor) que el paciente identifico o recordo correctamente. Pondera simple en el calculo de ORS. Debe ser menor o igual a `total_secondary_objects`.

---

**`incorrect_objects`** · `int >= 0` · Requerido

Cantidad de objetos que el paciente identifico incorrectamente (errores de reconocimiento o falsos positivos). Resta directamente del numerador de ORS, penalizando el desempeno.

---

**`total_key_objects`** · `int >= 0` · Requerido

Cantidad total de objetos **clave** que se evaluaron en la sesion. Pondera doble en el denominador de ORS. Si `total_key_objects + total_secondary_objects = 0`, ORS se fija en 0.

---

**`total_secondary_objects`** · `int >= 0` · Requerido

Cantidad total de objetos **secundarios** que se evaluaron en la sesion. Pondera simple en el denominador de ORS.

---

**Calculo de ORS (Object Recall Score)**

```
ORS = ((correct_key_objects * 2) + (correct_secondary_objects * 1) - (incorrect_objects * 1))
      ─────────────────────────────────────────────────────────────────────────────────────────
                  (total_key_objects * 2) + (total_secondary_objects * 1)
```

ORS puede ser negativo cuando los errores superan los aciertos ponderados. El modelo y la persistencia soportan rango ampliado.

---

**`total_events`** · `int >= 0` · Requerido

Cantidad total de eventos narrativos que ocurrieron durante la sesion VR. Actua como denominador en el calculo de ERS (Event Recall Score). Si se envia 0, ERS se fija en 0.

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

### Response

```json
{
  "metrics": { "ors": 0.667, "ers": 0.800, "scs": 1.0, "rta": 2.190, "ats": 0.800, "er": 0.200, "sps": 0.800 },
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
| `cognitive_level` | Nivel cognitivo de la sesion derivado deterministicamente del SPS (informativo para el terapeuta) |
| `recommendation` | Recomendacion de dificultad producida por el ML stateful |
| `probabilities` | Confianza del modelo en cada clase (suma 1.0). Permite trazabilidad clinica |
| `context` | Las 9 features de historial que el ML consumio + flag `cold_start` |
