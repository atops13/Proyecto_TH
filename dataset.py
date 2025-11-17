# --- dataset.py ---
import torch
from torch.utils.data import Dataset
import config 
import os

class AcousticDataset(Dataset):
    """
    Dataset de PyTorch que carga un archivo de features pre-procesados.
    """
    def __init__(self, features_file):
        """
        El constructor carga los datos desde el archivo .pt especificado.
        """
        if not os.path.exists(features_file):
            print(f"Error: No se encontró el archivo {features_file}.")
            print("Por favor, ejecuta primero 'preprocesado.py' para crearlo.")
            self.features = torch.empty(0, config.N_FEATURES)
            self.labels = torch.empty(0, dtype=torch.long)
        else:
            try:
                self.features, self.labels = torch.load(features_file)
                print(f"Dataset cargado desde {features_file}, {len(self.labels)} muestras encontradas.")
            except Exception as e:
                print(f"Error cargando el dataset {features_file}: {e}")
                self.features = torch.empty(0, config.N_FEATURES)
                self.labels = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        label = self.labels[idx]
        return feature_vector, label

# --- Código de prueba ---
if __name__ == "__main__":
    print("--- Probando AcousticDataset ---")
    
    dataset = AcousticDataset(config.TRAIN_FILE)
    
    if len(dataset) > 0:
        features, label = dataset[0]
        print("\nEjemplo de la primera muestra de TRAIN:")
        print(f"  Features: {features.shape} (Esperado: [{config.N_FEATURES}])")
        print(f"  Etiqueta: {label.item()} (Debe ser un número entre 0 y {config.N_CLASSES - 1})")
    else:
        print("Dataset de entrenamiento (TRAIN_FILE) vacío o no encontrado.")
        print("Por favor, ejecuta 'preprocesado.py' primero.")