import skops.io as sio
from datetime import datetime
from sklearn.metrics import classification_report
from .decoradores import controlTiempo

class Modelo:
    def __init__(self, nombre, ruta, log_path="logs/performance_log.txt", preds_path="Model_data/svm_preds.txt"):
        self.nombre = nombre
        self.ruta = ruta
        self.log_path = log_path
        self.preds_path = preds_path

    @staticmethod
    @controlTiempo
    def comprobar_tipo_fichero(**kwargs):
        vector_types = sio.get_untrusted_types(**kwargs)
        return vector_types
    
    @staticmethod
    @controlTiempo
    def cargar_modelo(VECTOR_MACHINE, vector_types):
        modelVectorMachine = sio.load(VECTOR_MACHINE, trusted=vector_types)
        return modelVectorMachine
    
    @staticmethod
    @controlTiempo
    def comprobar_modelo(X_test_scaled, modelVectorMachine):
        svm_preds = modelVectorMachine.predict(X_test_scaled)
        return svm_preds
    
    def registrar_inicio(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"INICIO EJECUCIÓN: {now} {self.nombre}\n")
            f.write("="*50 + "\n\n")
    
    def resgistrar_svm_preds(self, svm_preds, y_test):
         now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
         with open(self.preds_path, "a") as f:
            f.write(f"[{now}]Support Vector Machine Results:\n{classification_report(y_test, svm_preds)}\n\n")