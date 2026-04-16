import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import skops.io as sio
from .decoradores import controlTiempo, datetime, time, pasar_a_txt, log_exceptions
import logging
import os

os.makedirs("logs/errores", exist_ok=True)

logging.basicConfig(
    filename="logs/errores/app_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# DATOS
# =========================
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
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('Model_data/data_log.txt', 'a') as f:
                f.write(f"[{now}] {model} Results:\n{classification_report(y_test, preds)}\n\n")
        except Exception:
            logging.exception("Error al registrar predicciones")


# =========================
# OPERACIONES ML
# =========================
class Operaciones:

    @staticmethod
    @controlTiempo
    def procesamiento_datos(wine_df, wine_data):
        X = wine_df[wine_data.feature_names].copy()
        y = wine_df["target"].copy()
        return X, y

    @staticmethod
    @controlTiempo
    @staticmethod
    def instanciar_escalas(X):
        scaler = StandardScaler()

        if hasattr(X, "values"):
            X = X.values

        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    @staticmethod
    @controlTiempo
    def instanciar_modelos():
        return (
            LogisticRegression(),
            SVC(),
            DecisionTreeClassifier()
        )

    @staticmethod
    @controlTiempo
    def entrenar_modelos(logreg, svm, tree, X_train, y_train):
        logreg.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        tree.fit(X_train, y_train)
        return logreg, svm, tree


# =========================
# MODELO SERIALIZADO
# =========================
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


# =========================
# PIPELINE
# =========================
def pipeline_entrenamiento():

    t0 = time.time()

    wine_data = CargarDatos.cargar_datos()
    wine_df = CargarDatos.carga_df(wine_data)

    wine_df["target"] = wine_data.target

    t1 = time.time()

    X, y = Operaciones.procesamiento_datos(wine_df, wine_data)

    X_scaled, scaler = Operaciones.instanciar_escalas(X)

    t2 = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.7, random_state=25
    )

    logreg, svm, tree = Operaciones.instanciar_modelos()

    logreg, svm, tree = Operaciones.entrenar_modelos(
        logreg, svm, tree,
        X_train, y_train
    )

    t3 = time.time()

    print(f"Carga datos: {t1 - t0:.4f}s")
    print(f"Preprocesado: {t2 - t1:.4f}s")
    print(f"Entrenamiento: {t3 - t2:.4f}s")

    pasar_a_txt(f"""
Carga datos: {t1 - t0:.4f}s
Preprocesado: {t2 - t1:.4f}s
Entrenamiento: {t3 - t2:.4f}s
""")

    return {
        "logreg": logreg,
        "svm": svm,
        "tree": tree
    }, X_test, y_test


# =========================
# MENÚ
# =========================
def main():

    modelo_cargado = None
    modelos_entrenados = None
    X_test = None
    y_test = None

    while True:

        print("\n1. Entrenar modelos")
        print("2. Cargar modelo entrenado")
        print("3. Evaluar modelo")
        print("4. Salir")

        opcion = input("\nSelecciona opción: ")

        match opcion:

            case "1":
                print("Entrenando modelos...")
                modelos_entrenados, X_test, y_test = pipeline_entrenamiento()
                print("Entrenamiento completado")

            case "2":
                ruta = input("Introduce ruta del modelo: ")
                vector_types = Modelo.comprobar_tipo_fichero(file=ruta)
                modelo_cargado = Modelo.cargar_modelo(ruta, vector_types)
                print("Modelo cargado correctamente")

            case "3":

                if modelos_entrenados is None and modelo_cargado is None:
                    print("No hay modelos disponibles")
                    continue

                print("Elige modelo: (1)logreg / (2)svm / (3)tree")
                try:
                    num = int(input("Modelo: "))
                    match num:
                        case 1:
                            nombre = 'logreg'
                        case 2:
                            nombre = 'svm'
                        case 3: 
                            nombre = 'tree'
                        case _:
                            print('Elección inválida.')
                except ValueError:
                    print("Debes introducir un número")
                    
                if modelos_entrenados and nombre in modelos_entrenados:
                    modelo = modelos_entrenados[nombre]
                    preds = modelo.predict(X_test)
                    print(classification_report(y_test, preds))

                elif modelo_cargado:
                    preds = modelo_cargado.predict(X_test)
                    print(classification_report(y_test, preds))

                else:
                    print("Modelo no encontrado")

            case "4":
                print("Saliendo...")
                break

            case _:
                print("Opción no válida")


if __name__ == "__main__":
    main()
'''#Cargamos los datos en memoria
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
pasar_a_txt(f"Duración total ejecución: {total:.6f} segundos.")'''

