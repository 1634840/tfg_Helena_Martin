import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json 

# CONFIGURACIÓ
DATA_DIR = r"C:\\Users\\Asus\\Desktop\\TFG\\Data TFG UAB\\Data"
LABELS = {
    "supina_neutra": 0,
    "lateral_dreta": 1,
    "lateral_esquerra": 2,
    "sedestacio": 3,
    "inclinat": 4,
    "folwer": 5
}
KEYWORDS = {
    "supina_neutra": ["Supi_Neutre", "Supi_Talons"],
    "lateral_dreta": ["lateral_Dreta"],
    "lateral_esquerra": ["lateral_Esquerra"],
    "sedestacio": ["Sedestacio_Completa"],
    "inclinat": ["V.Dret 45", "V.Dret 30", "V a 22+22", "Supi_V.D 45"],
    "folwer": ["Low Fowler", "Semi-Fowler", "Full Fowler"]
}

transform = transforms.Compose([
    transforms.Resize((24, 32)),  # mida uniforme
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class PressioDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.paths = []
        self.transform = transform
        num_descartats = 0

        for subfolder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, subfolder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith("_Pressio.csv"):
                        label = self.get_label(file)
                        if label is None:
                            logging.warning(f"Etiqueta no trobada per fitxer: {file}")
                            num_descartats += 1
                            continue

                        csv_path = os.path.join(folder_path, file)
                        try:
                            arr = np.loadtxt(csv_path, delimiter=",", dtype=float)
                            arr = np.clip(arr, 0, 255)
                            if arr.max() == 0:
                                logging.warning(f"Fitxer buit (arr.max() == 0): {csv_path}")
                                num_descartats += 1
                                continue

                            arr = (arr / arr.max() * 255).astype(np.uint8)
                            img = Image.fromarray(arr)

                            if self.transform:
                                img = self.transform(img)

                            self.data.append(img)
                            self.labels.append(label)
                            self.paths.append(file)  


                        except Exception as e:
                            logging.error(f"Error carregant {csv_path}: {e}")
                            num_descartats += 1

        logging.info(f"Total mostres carregades: {len(self.data)}")
        
        logging.info(f"Total mostres descartades: {num_descartats}")
        assert len(self.data) == len(self.labels), "Nombre d'imatges i etiquetes no coincideixen"

    def get_label(self, filename):
        for class_name, keywords in KEYWORDS.items():
            if any(k in filename for k in keywords):
                return LABELS[class_name]
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = self.data[idx]
            label = self.labels[idx]
            if img.ndim != 3 or img.shape[0] != 1:
                raise ValueError(f"Imatge mal formada: {img.shape}")
            return img, label
        except Exception as e:
            logging.warning(f"[__getitem__] Error amb idx {idx}: {e}")
            return None

# VALIDACIÓ I PREPARACIÓ DE DADES VÀLIDES
raw_dataset = PressioDataset(DATA_DIR, transform=transform)

valid_samples = []
for i in range(len(raw_dataset)):
    try:
        sample = raw_dataset[i]
        if sample is None:
            continue
        img, label = sample
        path = raw_dataset.paths[i]
        if isinstance(img, torch.Tensor) and img.ndim == 3 and img.shape[0] == 1:
            valid_samples.append((img, label))
            print(path)
        else:
            logging.warning(f"Mostra descartada per forma incorrecta: idx {i}, shape = {img.shape}")
    except Exception as e:
        logging.warning(f"Error inesperat amb la mostra {i}: {e}")

logging.info(f"Total mostres vàlides: {len(valid_samples)}")

class PressioDatasetFiltrat(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = PressioDatasetFiltrat(valid_samples)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False)

print('val size', val_size)
print('train size', train_size)
# MODEL CNN 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(self._get_flattened_size(), 128)
        self.fc2 = nn.Linear(128, len(LABELS))

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 24, 32)  # mida després de Resize
            x = self.pool(torch.relu(self.conv1(dummy)))
            x = self.pool(torch.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ENTRENAMENT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        if outputs.shape[0] != labels.shape[0]:
            logging.error(f"Desajust batch: outputs={outputs.shape[0]}, labels={labels.shape[0]} — saltant batch")
            continue

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")

# VALIDACIÓ



model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\\nAccuracy sobre el set de validació: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "model_postures_pressio.pth")



all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS.keys())
fig, ax = plt.subplots(figsize=(8, 6))  

disp.plot(ax=ax, cmap="Blues", xticks_rotation='vertical')
plt.tight_layout()  
plt.show()


misclassified = []

model.eval()
with torch.no_grad():
    for idx, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                # Índex absolut dins del dataset original
                global_idx = val_dataset.indices[idx * val_loader.batch_size + i]
                filename = raw_dataset.paths[global_idx]
                true_class = [k for k, v in LABELS.items() if v == labels[i].item()][0]
                pred_class = [k for k, v in LABELS.items() if v == predicted[i].item()][0]
                misclassified.append((filename, true_class, pred_class))

# Mostrar resultats
print("\n Errors de classificació:")
for file, real, pred in misclassified:
    print(f"- {file}: esperat = {real}, predit = {pred}")


