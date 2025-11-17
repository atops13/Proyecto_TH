# --- preprocesado.py ---
import os
import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 
import config

try:
    from tqdm import tqdm
except ImportError:
    print("Nota: 'tqdm' no está instalado. Ejecuta 'pip install tqdm' para ver barras de progreso.")
    def tqdm(iterable, *args, **kwargs):
        return iterable

def extract_features(y, sr):
    """
    Extrae un vector de 16 características de un *segmento* de audio (y).
    """
    try:
        rms = librosa.feature.rms(y=y, frame_length=config.FRAME_SIZE, hop_length=config.HOP_LENGTH)
        mean_rms = np.mean(rms)
        
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=config.FRAME_SIZE, hop_length=config.HOP_LENGTH)
        mean_zcr = np.mean(zcr)
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=config.FRAME_SIZE, hop_length=config.HOP_LENGTH)
        mean_centroid = np.mean(centroid)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.FRAME_SIZE, hop_length=config.HOP_LENGTH)
        mean_mfccs = np.mean(mfccs, axis=1)
        
        feature_vector = np.concatenate((
            np.array([mean_rms]),
            np.array([mean_zcr]),
            np.array([mean_centroid]),
            mean_mfccs
        )).astype(np.float32)
        
        if np.isnan(feature_vector).any():
            return None 
            
        return feature_vector

    except Exception as e:
        print(f"Error al extraer features: {e}")
        return None

def process_and_split_dataset():
    """
    Recorre 'data/', segmenta los .wav, extrae features,
    divide en Train/Val/Test, normaliza y guarda 3 archivos .pt
    """
    all_features = []
    all_labels = []
    
    print("Iniciando preprocesado y segmentación...")
    
    for label_idx, class_name in config.CLASSES_MAP.items():
        class_dir = os.path.join(config.DATA_DIR, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"ADVERTENCIA: No se encontró la carpeta: {class_dir}")
            continue
            
        print(f"Procesando clase: {class_name} (Etiqueta: {label_idx})")
        
        for file_name in tqdm(os.listdir(class_dir)):
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                
                try:
                    y_full, sr = librosa.load(file_path, sr=config.SAMPLING_RATE, mono=True)
                    
                    num_segments = len(y_full) // config.SAMPLES_PER_SEGMENT
                    
                    if num_segments == 0:
                        print(f"  Advertencia: {file_name} es más corto que {config.SEGMENT_LENGTH_SEC}s. Saltando.")
                        continue

                    for i in range(num_segments):
                        start_sample = i * config.SAMPLES_PER_SEGMENT
                        end_sample = start_sample + config.SAMPLES_PER_SEGMENT
                        y_segment = y_full[start_sample:end_sample]
                        
                        features = extract_features(y_segment, sr)
                        
                        if features is not None:
                            all_features.append(features)
                            all_labels.append(label_idx) 
                            
                except Exception as e:
                    print(f"Error cargando o segmentando {file_path}: {e}")

    if len(all_features) == 0:
        print("\n¡ERROR CRÍTICO! No se procesó ningún archivo .wav o todos eran muy cortos.")
        print(f"Verifica que la carpeta '{config.DATA_DIR}/' contenga las subcarpetas:")
        print(f"{list(config.CLASSES_MAP.values())}")
        return

    print(f"\nTotal de {len(all_features)} segmentos de {config.SEGMENT_LENGTH_SEC}s extraídos.")

    X = np.array(all_features)
    y = np.array(all_labels)

    # 1. Separar Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SPLIT_SIZE, 
        random_state=config.RANDOM_SEED,
        stratify=y 
    )

    # 2. Calcular % de validación
    val_size_relative = config.VALIDATION_SPLIT_SIZE / (1.0 - config.TEST_SPLIT_SIZE)
    
    # 3. Separar Train y Validation del resto
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size_relative, 
        random_state=config.RANDOM_SEED,
        stratify=y_train_val 
    )

    print(f"División de datos completada:")
    print(f"  Entrenamiento: {len(X_train)} muestras")
    print(f"  Validación:    {len(X_val)} muestras")
    print(f"  Test:          {len(X_test)} muestras")

    # --- Normalización de Features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Features normalizados (StandardScaler).")

    # --- Guardado de Archivos ---
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    
    torch.save((torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), config.TRAIN_FILE)
    torch.save((torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), config.VAL_FILE)
    torch.save((torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), config.TEST_FILE)
    
    joblib.dump(scaler, config.SCALER_FILE)
    
    print(f"Archivos de datos (train, val, test) guardados en '{config.FEATURES_DIR}/'")
    print(f"Scaler guardado en '{config.SCALER_FILE}'")


if __name__ == "__main__":
    process_and_split_dataset()