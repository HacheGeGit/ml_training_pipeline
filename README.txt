# ML Training Pipeline

Proyecto de ejemplo para entrenar, evaluar y guardar modelos de machine learning usando `scikit-learn` y `skops.io`.  
Permite trabajar con cualquier dataset compatible (actualmente se usa el dataset Wine de sklearn), entrenar varios modelos, registrar tiempos de ejecución y guardar resultados de predicciones y datos escalados.

---

## Características

- Entrenamiento de modelos:  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Decision Tree  
- Registro de tiempos de ejecución de cada tarea mediante decoradores.  
- Guardado de modelos entrenados en formato '.skops'.  
- Guardado de resultados y predicciones por modelo en archivos separados.  
- Almacenamiento de datasets escalados para análisis posterior.  
- Estructura modular y testeable.
- Tests automatizados con pytest.

---

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/tu_usuario/ml_training_pipeline.git
cd ml_training_pipeline
pip install -e .

Esto instalará:
    -El paquete pipeline_pkg
    -Todas las dependencias definidas en pyproject.toml

## Uso
Hacer which python en terminal.
Si python3:
python3 -m pipeline_pkg.pipeline
Si pyhton:
python -m pipeline_pkg.pipeline

El flujo de ejecución es el siguiente:
    -Carga de datos
    -Conversión a DataFrame
    -Procesamiento y escalado
    -Entrenamiento de modelos
    -Evaluación y predicciones
    -Persistencia de modelos
    -Guardado de resultados y logs

## Estructura del proyecto:
ml_training_pipeline/          # Carpeta raíz del proyecto
│
├─ pipeline_pkg/               # Paquete Python con la lógica de ML
│   ├─ __init__.py
│   ├─ pipeline.py             # Script principal con clases y ejecución
│   └─ decoradores.py          # Decoradores de control de tiempo y funciones auxiliares
│
├─ pyproject.toml              # Configuración del paquete (instalable con pip)
├─ pytest.ini                  # Configuración de pytest
├─ README.md                   # Documentación del proyecto
│
├─ DataFrames/                 # CSVs de datos procesados
├─ Model_data/                 # Modelos entrenados y predicciones
├─ logs/                       # Registros de ejecución y tiempos
└─ skopsModels/                # Modelos guardados con skops

Dependencias principales:
    -scikit-learn
    -pandas
    -numpy
    -skops

## Testing

Ejecutar los tests con:

```bash
pytest

## Licencia:

Este proyecto está bajo licencia MIT. Puedes usarlo, modificarlo y redistribuirlo libremente.