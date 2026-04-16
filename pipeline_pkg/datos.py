import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from .decoradores import controlTiempo, datetime, log_exceptions
import logging
import os

class CargarDatos:

    @staticmethod
    @log_exceptions
    @controlTiempo
    def cargar_datos():
        return load_wine()

    @staticmethod
    @log_exceptions
    @controlTiempo
    def carga_df(wine_data):
        return pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

    @staticmethod    
    def registrar_predicciones(model, y_test, preds):
        try:
            os.makedirs("Model_data", exist_ok=True)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("Model_data/data_log.txt", "a") as f:
                f.write(f"\n[{now}]{model} Results:\n")
                f.write(classification_report(y_test, preds))
                f.write("\n")

        except Exception:
            logging.exception("Error al registrar predicciones")