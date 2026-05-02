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