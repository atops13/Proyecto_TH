# --- evaluacion.py ---
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config
from dataset import AcousticDataset
from model import AEC_MLP

def evaluate_model():
    """
    Carga el modelo entrenado y lo evalúa contra el Test Set.
    Genera la Matriz de Confusión y el Informe de Clasificación.
    """
    
    print("--- Iniciando Evaluación Final ---")
    
    # --- 1. Cargar Dataset de Test ---
    try:
        test_dataset = AcousticDataset(config.TEST_FILE)
        if len(test_dataset) == 0:
            raise RuntimeError("Test dataset está vacío.")
    except Exception as e:
        print(f"Error fatal: No se pudo cargar el test set. {e}")
        print("Asegúrate de haber ejecutado 'preprocesado.py' con éxito primero.")
        return
        
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Usando {len(test_dataset)} muestras de Test.")

    # --- 2. Cargar el Modelo Entrenado ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AEC_MLP().to(device)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo '{config.MODEL_SAVE_PATH}'.")
        print("Asegúrate de haber ejecutado 'entenamiento.py' primero.")
        return
        
    model.eval() 

    # --- 3. Obtener Predicciones ---
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # --- 4. Generar Métricas (Respuesta a Carmen) ---
    
    # Nombres de las clases para los informes (base 0)
    unique_labels_0_indexed = sorted(list(set(all_targets)))
    
    if not unique_labels_0_indexed:
        print("Error: No se encontraron etiquetas en el test set.")
        return
        
    class_names = [config.CLASSES_MAP[label] for label in unique_labels_0_indexed if label in config.CLASSES_MAP]

    # a) Informe de Clasificación
    print("\n--- Informe de Clasificación (Fiabilidad por Clase) ---")
    report = classification_report(
        all_targets, 
        all_predictions, 
        labels=unique_labels_0_indexed, 
        target_names=class_names,
        zero_division=0 # Evita warnings si una clase no tiene predicciones
    )
    print(report)

    # b) Matriz de Confusión
    print("\n--- Matriz de Confusión ---")
    cm = confusion_matrix(all_targets, all_predictions, labels=unique_labels_0_indexed)
    print(cm)
    
    # c) Graficar la Matriz de Confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión - Test Set')
    plt.savefig('matriz_confusion.png')
    print("\nMatriz de confusión guardada en 'matriz_confusion.png'")
    # plt.show() # Descomenta si quieres que la gráfica aparezca al final

if __name__ == "__main__":
    evaluate_model()