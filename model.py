# --- model.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import config 

class AEC_MLP(nn.Module):
    def __init__(self):
        """
        Define la arquitectura del Perceptrón Multicapa (MLP).
        """
        super(AEC_MLP, self).__init__()
        
        self.fc1 = nn.Linear(config.N_FEATURES, 15)
        self.fc2 = nn.Linear(15, config.N_CLASSES)

    def forward(self, x):
        """
        Define el paso "hacia adelante" de la red.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Código de prueba (para ejecutar solo este archivo) ---
if __name__ == "__main__":
    print("--- Probando la arquitectura del modelo ---")
    
    dummy_input = torch.randn(4, config.N_FEATURES)
    print(f"Forma de la entrada: {dummy_input.shape}")
    
    model = AEC_MLP()
    print("\nArquitectura del modelo:")
    print(model)
    
    output = model(dummy_input)
    print(f"\nForma de la salida: {output.shape} (Esperado: [4, {config.N_CLASSES}])")