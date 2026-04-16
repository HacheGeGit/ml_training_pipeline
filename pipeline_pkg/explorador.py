from tkinter import Tk
from tkinter.filedialog import askopenfilename

def seleccionar_modelo():
    root = Tk()
    root.withdraw()  # oculta ventana principal
    root.attributes("-topmost", True)
    root.update()

    ruta = askopenfilename(
        title="Selecciona un modelo",
        filetypes=[("Skops models", "*.skops"), ("Todos los archivos", "*.*")]
    )

    root.destroy()
    return ruta