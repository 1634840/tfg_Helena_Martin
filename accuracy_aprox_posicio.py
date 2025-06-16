import os
import json

# Ruta al JSON amb les prediccions
prediccions_path = r"C:\Users\Asus\Desktop\TFG\postures\postures_totals.json"

# Keywords i mapping d’etiquetes reals (pel nom del fitxer)
KEYWORDS = {
    "supina_neutra": ["Supi_Neutre"],
    "lateral_dreta": ["lateral_Dreta"],
    "lateral_esquerra": ["lateral_Esquerra"],
    "sedestacio": ["Sedestacio_Completa", "Supi_Talons"],
    "inclinat": ["V.Dret 45", "V.Dret 30", "V a 22+22", "Supi_V.D 45"],
    "folwer": ["Low Fowler", "Semi-Fowler", "Full Fowler"]
}

LABELS = {
    "supina_neutra": 2,
    "lateral_dreta": 1,
    "lateral_esquerra": 0,
    "sedestacio": 3,
    "inclinat": 4,
    "folwer": 5
}

def infer_label_from_filename(filename):
    parts = filename.split("_", 2)
    if len(parts) < 3:
        return None
    name_part = parts[2]  # La part rellevant

    for label, keywords in KEYWORDS.items():
        for kw in keywords:
            if kw in name_part:
                return LABELS[label]
    return None

# 🔹 Carrega les prediccions
with open(prediccions_path, "r") as f:
    prediccions = json.load(f)

# 🔍 Comparació
total = 0
correct = 0
known_total = 0
known_correct = 0

for entry in prediccions:
    filename = entry["file_name"]
    predicted = entry["position"]
    real = infer_label_from_filename(filename)

    if real is None:
        continue

    total += 1
    if predicted == real:
        correct += 1

    if real in [0, 1, 2]:  # només supina/laterals
        known_total += 1
        if predicted == real:
            known_correct += 1
        else:
            print(f"❌ {filename} — Predicció: {predicted} | Real: {real}")

# 📊 Resultats
acc_total = correct / total if total else 0
acc_known = known_correct / known_total if known_total else 0

print(f"\nAccuracy total: {acc_total:.2%}")
print(f"Accuracy només supina/laterals: {acc_known:.2%}")
