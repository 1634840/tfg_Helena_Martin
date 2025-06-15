# The code defines a PyTorch model that combines a DenseNet with joint data for a posture
# classification task, including dataset loading, model training, testing, and validation with
# confusion matrix visualization.
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CONFIGURACIÓ
JSONL_PATH = r"C:\Users\Asus\Desktop\TFG\dataset_etiquetat.jsonl"
#JSONL_PATH = "dataset_filtrat_roba_etiquetes.jsonl"
IMAGE_SIZE = (64, 64)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

LABELS = {
    0: "supina_neutra",
    1: "lateral_dreta",
    2: "lateral_esquerra",
    3: "sedestacio",
    4: "inclinat_dret",
    5: "folwer"
}

# Dataset
class PosturaAmbArticulacionsDataset(Dataset):
    def __init__(self, jsonl_path, transform=None):
        with open(jsonl_path, "r") as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        csv_path = entry["csv_path"]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ No s'ha trobat el fitxer de pressió: {csv_path}")
        img_array = np.loadtxt(csv_path, delimiter=",", dtype=float)
        img_array = np.clip(img_array, 0, 255)
        if img_array.max() > 0:
            img_array = (img_array / img_array.max()) * 255
        img = Image.fromarray(img_array.astype(np.uint8)).convert("L")

        if self.transform:
            img = self.transform(img)

        joints = torch.tensor(entry["input"], dtype=torch.float32)
        label = torch.tensor(entry["label"], dtype=torch.long)
        return img, joints, label

# Model: ResNet + articulacions
class FusionResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.resnet = resnet

        self.joint_net = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, img, joints):
        x_img = self.resnet(img)
        x_joint = self.joint_net(joints)
        x = torch.cat((x_img, x_joint), dim=1)
        return self.classifier(x)

class FusionDenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet121(weights=None)
        
        # Modifiquem la primera capa perquè accepti imatges en escala de grisos (1 canal)
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet = densenet
        
        # La sortida de DenseNet121 és de mida 1024
        self.joint_net = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 5 classes
        )

    def forward(self, img, joints):
        x_img = self.densenet.features(img)
        x_img = nn.functional.adaptive_avg_pool2d(x_img, (1, 1)).view(img.size(0), -1)
        x_joint = self.joint_net(joints)
        x = torch.cat((x_img, x_joint), dim=1)
        return self.classifier(x)

# Entrenament
dataset = PosturaAmbArticulacionsDataset(JSONL_PATH, transform=transform)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
print('test size', test_size)
print('train size', train_size)
print('val size', val_size)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionResNetModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0
    for imgs, joints, labels in train_loader:
        imgs, joints, labels = imgs.to(device), joints.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, joints)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")

# Test i matriu confusió
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, joints, labels in test_loader:
        imgs, joints = imgs.to(device), joints.to(device)
        outputs = model(imgs, joints)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\\n🎯 Test Accuracy: {acc*100:.2f}%")

# VALIDACIÓ (després d’entrenar)
model.eval()
val_correct = 0
val_total = 0
with torch.no_grad():
    for imgs, joints, labels in val_loader:
        imgs, joints = imgs.to(device), joints.to(device)
        outputs = model(imgs, joints)
        _, preds = torch.max(outputs, 1)
        val_correct += (preds.cpu() == labels).sum().item()
        val_total += labels.size(0)

val_acc = 100 * val_correct / val_total
print(f"✅ Validació Accuracy: {val_acc:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[LABELS[i] for i in range(6)])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues")  # cmap ha d'anar aquí si suportat
plt.setp(ax.get_xticklabels(), rotation=45)  # rotació dels ticks
plt.title("Matriu de confusió (Test)")
plt.tight_layout()
plt.show()
