import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import skops.io as sio
from .decoradores import controlTiempo, datetime, time, pasar_a_txt

class CargarDatos:
    def __init__(self, datos):
        self.datos = datos

    @staticmethod
    @controlTiempo
    def cargar_datos():
         wine_data = load_wine()
         return wine_data
    
    @staticmethod
    @controlTiempo
    def carga_df(wine_data):
        wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
        return wine_df
    @staticmethod    
    def registrar_predicciones(model, y_test, preds):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('Model_data/data_log.txt', 'a') as f:
            f.write(f"[{now}]{model} Results:\n{classification_report(y_test, preds)}\n\n")
        
class Operaciones:
    @staticmethod
    @controlTiempo
    def mostrar_info(wine_df):
        wine_df.info()
        wine_df.describe()
        wine_df.head()
        wine_df.tail()
    
    @staticmethod
    @controlTiempo
    def procesamiento_datos(wine_df):
        X = wine_df[wine_data.feature_names].copy()
        y = wine_df["target"].copy()  
        return X, y
    
    @staticmethod
    @controlTiempo
    def instanciar_escalas(X):
        scaler = StandardScaler()
        scaler.fit(X)
        # Transformamos las características
        X_scaled = scaler.transform(X.values)
        return X_scaled
    
    @staticmethod
    @controlTiempo
    def instanciar_modelos():
        logistic_regression = LogisticRegression() 
        svm = SVC() 
        tree = DecisionTreeClassifier()
        return logistic_regression, svm, tree
    
    @staticmethod
    @controlTiempo
    def entrenar_modelos(logistic_regression, svm, tree):
        logistic_regression.fit(X_train_scaled, y_train) 
        svm.fit(X_train_scaled, y_train) 
        tree.fit(X_train_scaled, y_train)
        return logistic_regression, svm, tree
    
    @staticmethod
    @controlTiempo
    def hacer_predicciones(X_test_scaled):
        log_reg_preds = logistic_regression.predict(X_test_scaled) 
        svm_preds = svm.predict(X_test_scaled) 
        tree_preds = tree.predict(X_test_scaled)
        return log_reg_preds, svm_preds, tree_preds
    
    @staticmethod
    @controlTiempo
    def guardar_modelo(modelo,ruta):
        sio.dump(modelo, ruta)
        return ruta

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
        
modelo_wine = Modelo("Modelo Wine","./skopsModels/VectorMachine.skops")   
modelo_wine.registrar_inicio()    
start_total = time.time()

#Cargamos los datos en memoria
wine_data = CargarDatos.cargar_datos()

# Convertimos los datos de wine_data a pandas dataframe
wine_df = CargarDatos.carga_df(wine_data)

# Añadimos la table con los datos
wine_df["target"] = wine_data.target
wine_df.to_csv('DataFrames/winedf.csv', index=False)
# Previsualizamos los datos
Operaciones.mostrar_info(wine_df)

#Procesamiento de datos
# Dividimos los datos en características y etiquetas
X, y = Operaciones.procesamiento_datos(wine_df)

# Instanciamos las escalas y ajustamos las características
# Transformamos las características
X_scaled = Operaciones.instanciar_escalas(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.to_csv('DataFrames/X_scaled.csv', index=False)

# Comprobamos la primera instancia
print(X_scaled[0])
"""
[ 1.51861254 -0.5622498 0.23205254 -1.16959318 1.91390522 0.80899739
1.03481896 -0.65956311 1.22488398 0.25171685 0.36217728 1.84791957
1.01300893]
"""

# Dividimos los datos de entrenamiento y de prueba
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,y,train_size=.7, random_state=25)

# Comprobamos que la division se ha realizado correctamente
print(f"Train size: {round(len(X_train_scaled) / len(X) * 100)}% \n\
Test size: {round(len(X_test_scaled) / len(X) * 100)}%")
""" Result del print:
Train size: 70%
Test size: 30%
"""
print()

# Instanciamos los modelos 
logistic_regression, svm, tree = Operaciones.instanciar_modelos()

# Entrenamos los modelos
Operaciones.entrenar_modelos(logistic_regression, svm, tree)

# Hacemos predicciones con cada modelo 
log_reg_preds, svm_preds, tree_preds = Operaciones.hacer_predicciones(X_test_scaled) 

# Almacenamos los modelos de predicción en un diccionario. 
# Esto hace más fácil la iteración de cada modelo e imprimir los resultados.
model_preds = {
    "Logistic Regression": log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds
}

for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
    CargarDatos.registrar_predicciones(model, y_test, preds)


# Una vez entrenado y evaluado cualquier modelo, podemos guardarlo para su posterior uso o migración
# Para ello utilizaremos la librería skops.io y guardaremos el modelo en un fichero skops
ruta = modelo_wine.ruta
VECTOR_MACHINE = Operaciones.guardar_modelo(svm, ruta)

"""
****** Cargar de nuevo el modelo y comprobar que funciona correctamente
"""

# Investigar el tipo de fichero

vector_types = Modelo.comprobar_tipo_fichero(file=VECTOR_MACHINE)

# Y cargarlo solo si se confía en él

modelVectorMachine = Modelo.cargar_modelo(VECTOR_MACHINE, vector_types)

# Comprobamos el modelo después de cargarlo, vemos que con los mismos datos de prueba predice lo mismo que anteriormente

svm_preds = Modelo.comprobar_modelo(X_test_scaled, modelVectorMachine)

print(f"Support Vector Machine Results:\n{classification_report(y_test, svm_preds)}", sep="\n\n")
modelo_wine.resgistrar_svm_preds(svm_preds, y_test)
end_total = time.time()
total = end_total - start_total
print(f"\nDuración total ejecución: {total:.6f} segundos.\n")
pasar_a_txt(f"Duración total ejecución: {total:.6f} segundos.")