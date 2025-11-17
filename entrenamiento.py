# --- entenamiento.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import AEC_MLP
from dataset import AcousticDataset
import config

# --- 1. Configuración del dispositivo ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- 2. Carga de datos (ya divididos por preprocesado.py) ---
try:
    train_dataset = AcousticDataset(config.TRAIN_FILE)
    val_dataset = AcousticDataset(config.VAL_FILE)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Dataset de entrenamiento o validación están vacíos.")
except Exception as e:
    print(f"Error fatal: No se pudieron cargar los datos. {e}")
    print("Asegúrate de haber ejecutado 'preprocesado.py' con éxito primero.")
    exit() 

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)

print(f"Datos cargados: {len(train_dataset)} muestras de entrenamiento, {len(val_dataset)} de validación.")

# --- 3. Inicializar Modelo, Pérdida y Optimizador ---
model = AEC_MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

# Listas para almacenar métricas
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# --- 4. Función de Entrenamiento ---
def train(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(data)
    
    avg_loss = total_loss / total
    avg_accuracy = 100. * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    
    print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

# --- 5. Función de Validación ---
def validate():
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(data)

    val_loss /= total
    accuracy = 100. * correct / total
    
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)
    
    print(f'\nValidation set: Average loss: {val_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

# --- 6. Bucle Principal de Ejecución ---
print("Iniciando entrenamiento...")
best_accuracy = 0
for epoch in range(1, config.NUM_EPOCHS + 1):
    train(epoch)
    accuracy = validate()
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado en '{config.MODEL_SAVE_PATH}' con {accuracy:.2f}% de precisión")

print(f"\nEntrenamiento completado. Mejor precisión de validación: {best_accuracy:.2f}%")

# --- 7. Graficar Curvas ---
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
print("Curvas de entrenamiento guardadas en 'training_curves.png'")