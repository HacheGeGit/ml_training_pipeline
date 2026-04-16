import pandas as pd
from .operaciones import Operaciones
from .datos import CargarDatos
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import skops.io as sio
from .decoradores import controlTiempo, datetime, time, pasar_a_txt, log_exceptions

class Modelo:

    def __init__(self, nombre, ruta):
        self.nombre = nombre
        self.ruta = ruta

    @staticmethod
    @controlTiempo
    def comprobar_tipo_fichero(**kwargs):
        try:
            return sio.get_untrusted_types(**kwargs)
        except FileNotFoundError:
            return []

    @staticmethod
    @controlTiempo
    def cargar_modelo(ruta, vector_types):
        return sio.load(ruta, trusted=vector_types)