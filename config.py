# --- config.py ---
# Este archivo contiene todas las variables de configuración del proyecto
import os

# 1. Parámetros de Audio
SAMPLING_RATE = 8000     # Tasa de muestreo (Hz)
SEGMENT_LENGTH_SEC = 2   # Duración de cada muestra de audio (en segundos)
FRAME_SIZE = 1024        # Tamaño de la trama FFT
HOP_LENGTH = 512         # Solapamiento
N_MFCC = 13              # Número de coeficientes MFCC a extraer

# Cálculo automático de muestras por segmento
SAMPLES_PER_SEGMENT = SAMPLING_RATE * SEGMENT_LENGTH_SEC 

# 2. Parámetros del Modelo
N_FEATURES = 1 + 1 + 1 + N_MFCC  # Total: 16 (RMS, ZCR, Centroide, 13 MFCCs)
N_CLASSES = 4                    # Número de entornos a clasificar

# 3. Parámetros de Entrenamiento
BATCH_SIZE = 32
NUM_EPOCHS = 40                 
LEARNING_RATE = 0.01
RANDOM_SEED = 42                 # Semilla fija para que las divisiones sean siempre iguales

# --- LÍNEAS CORREGIDAS ---
TEST_SPLIT_SIZE = 0.15           # 15% para Test
VALIDATION_SPLIT_SIZE = 0.15     # 15% para Validación (del 85% restante)
# -------------------------

# 4. Rutas de Archivos
DATA_DIR = "./data"                     # Carpeta raíz con las subcarpetas de clases
FEATURES_DIR = "./features"             # Carpeta para guardar los datos procesados

# Rutas de archivos
TRAIN_FILE = os.path.join(FEATURES_DIR, "train_data.pt")
VAL_FILE = os.path.join(FEATURES_DIR, "val_data.pt")
TEST_FILE = os.path.join(FEATURES_DIR, "test_data.pt")
SCALER_FILE = os.path.join(FEATURES_DIR, "scaler.pkl") # Archivo para guardar el normalizador

MODEL_SAVE_PATH = "aec_best_model.pth"        # Nombre del modelo final guardado

# 5. Clases (en orden, base-0)
CLASSES_MAP = {
    0: "clase_1_silencio",
    1: "clase_2_trafico",
    2: "clase_3_evento",
    3: "clase_4_bar"
}