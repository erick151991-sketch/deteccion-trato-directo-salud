# Detección de trato directo en compras públicas de Salud


La idea del proyecto nació a partir de una pregunta simple: ¿es posible usar aprendizaje automático para identificar compras públicas que tengan características similares a un trato directo? Para explorarlo, trabajamos con órdenes de compra del sector Salud publicadas en Mercado Público y construimos un modelo que entrega una probabilidad de trato directo para cada registro.

El resultado debe entenderse como una herramienta de apoyo para priorizar revisiones. El modelo no determina que una compra sea irregular ni reemplaza el análisis de una persona.

## Objetivo

El objetivo principal es desarrollar un flujo completo de análisis de datos que permita:

- cargar y revisar el dataset original;
- limpiar y preparar la información;
- comparar distintos modelos de machine learning y deep learning;
- evaluar los resultados con datos no utilizados durante el entrenamiento;
- explicar las predicciones mediante SHAP;
- presentar el modelo a través de una aplicación web sencilla.

## Estructura del proyecto

```text
deteccion-trato-directo-salud/
├── data/
│   └── oc_salud_2026_08.csv
├── src/
│   ├── config.py
│   ├── preparacion.py
│   ├── eda.py
│   ├── modelado.py
│   ├── dl.py
│   ├── evaluacion.py
│   └── frecuencias.py
├── app/
│   ├── app.py
│   └── templates/
├── resultados/
│   ├── figuras/
│   ├── tablas/
│   └── modelos/
├── requirements.txt
├── EJECUTAR.bat
└── README.md
```

Los archivos de la carpeta `src` cumplen las siguientes funciones:

- `preparacion.py`: carga, limpieza, transformación y división de los datos.
- `eda.py`: análisis exploratorio, tablas y gráficos.
- `modelado.py`: entrenamiento y comparación de modelos clásicos.
- `dl.py`: entrenamiento de las redes neuronales.
- `evaluacion.py`: evaluación final, curvas, matriz de confusión, SHAP y K-Means.
- `frecuencias.py`: genera las tablas que necesita la aplicación web.

## Dataset

El dataset utilizado contiene órdenes de compra del sector Salud correspondientes a agosto de 2026. El archivo original tiene aproximadamente 137 mil filas y 25 columnas.

El CSV no está incluido directamente en el repositorio porque pesa cerca de 75 MB. Para ejecutar todo el proyecto se debe copiar con el siguiente nombre y ubicación:

```text
data/oc_salud_2026_08.csv
```

El archivo utiliza punto y coma (`;`) como separador.

Durante la preparación se excluyen las compras ágiles y los convenios marco. También se eliminan del modelo las columnas que revelan directamente el mecanismo de compra, ya que producirían fuga de información.

## Requisitos

Para ejecutar el proyecto se necesita:

- Python 3.10 o superior;
- Git, si se desea clonar el repositorio;
- aproximadamente 2 GB de espacio disponible;
- conexión a internet durante la instalación de dependencias.

Las bibliotecas necesarias están declaradas en `requirements.txt`.

## Descargar el proyecto

El repositorio se puede clonar con:

```bash
git clone https://github.com/erick151991-sketch/deteccion-trato-directo-salud.git
cd deteccion-trato-directo-salud
```

También se puede descargar desde GitHub usando **Code > Download ZIP**.

## Ejecución rápida en Windows

La forma más simple de ejecutar todo el procesamiento es:

1. Descargar o clonar el repositorio.
2. Copiar el dataset en `data/oc_salud_2026_08.csv`.
3. Ejecutar `EJECUTAR.bat` con doble clic.

El archivo crea un entorno virtual, instala las dependencias y ejecuta las etapas del proyecto en orden. El detalle de la ejecución queda guardado en `log_ejecucion.txt`.

La primera ejecución puede tardar varios minutos porque se instalan TensorFlow, SHAP y las demás dependencias.

## Instalación manual

### Windows

Abrir PowerShell dentro de la carpeta del proyecto y ejecutar:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux, macOS o GitHub Codespaces

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecutar el análisis completo

Con el entorno activado y el dataset ubicado en la carpeta `data`, ejecutar:

```bash
python src/eda.py
python src/modelado.py
python src/evaluacion.py
python src/frecuencias.py
```

Cada comando realiza una etapa diferente:

1. `eda.py` revisa la calidad de los datos y genera el análisis exploratorio.
2. `modelado.py` prepara los datos, entrena los modelos y compara sus métricas.
3. `evaluacion.py` utiliza el conjunto de prueba para la evaluación final y genera las explicaciones SHAP.
4. `frecuencias.py` crea la información adicional utilizada por el formulario web.

Los resultados se guardan automáticamente en:

```text
resultados/figuras/
resultados/tablas/
resultados/modelos/
```

## Ejecutar solamente la aplicación web

Una vez que existen el preprocesador y el modelo entrenado, ejecutar:

```bash
python app/app.py
```

Luego abrir en el navegador:

```text
http://127.0.0.1:5000/
```

La aplicación incluye las siguientes rutas:

| Ruta | Descripción |
|---|---|
| `/` | Formulario para evaluar una compra. |
| `/monitoreo` | Panel de uso y monitoreo del modelo. |
| `/salud` | Comprueba si el servicio está funcionando. |
| `/api/evaluar` | API que recibe datos en formato JSON. |

Ejemplo de consulta a la API:

```bash
curl -X POST http://127.0.0.1:5000/api/evaluar \
  -H "Content-Type: application/json" \
  -d '{"MontoTotalOC_PesosChilenos":12000000,"precioNeto":850000,"cantidad":2}'
```

## Preparación de los datos

Antes del entrenamiento se realizan, entre otras, las siguientes tareas:

- conversión de variables monetarias a formato numérico;
- normalización de nombres de regiones;
- imputación de datos faltantes;
- transformaciones logarítmicas para montos y cantidades;
- creación de razones e indicadores de ausencia;
- extracción del día y día de la semana desde la fecha;
- codificación one-hot de variables categóricas;
- codificación por frecuencia para categorías con muchos valores;
- estandarización de variables numéricas;
- separación por orden de compra en entrenamiento, validación y prueba.

La separación se realiza usando el código de la orden. De esta forma, dos líneas pertenecientes a la misma compra no quedan repartidas entre entrenamiento y prueba.

## Modelos evaluados

En el proyecto se compararon tres modelos clásicos:

- regresión logística;
- bosque aleatorio;
- gradient boosting.

Cada uno se probó sin balanceo, con ponderación de clases, con SMOTE y con submuestreo. Además, se entrenaron tres redes neuronales:

- MLP simple;
- MLP profunda;
- red con conexiones residuales.

La métrica principal fue AUC-PR, ya que la clase de trato directo representa menos del 10 % de los registros y existe un desbalance importante.

## Resultado principal

El modelo con mejor rendimiento fue `HistGradientBoostingClassifier` sin balanceo. En el conjunto de prueba obtuvo aproximadamente:

| Métrica | Resultado |
|---|---:|
| AUC-ROC | 0,938 |
| AUC-PR | 0,741 |
| Precisión | 0,696 |
| Recall | 0,626 |
| F1 | 0,659 |

Para estas métricas se utilizó un umbral cercano a `0,252`, seleccionado con el conjunto de validación.

## Consideraciones y limitaciones

La etiqueta utilizada indica el mecanismo administrativo de compra y no la existencia de fraude o incumplimiento. Por eso, el resultado del modelo solo representa similitud con el perfil observado de los tratos directos.

## Integrantes

- Luis Almarza
- Nicolas Tobar
- Eric Amigo


