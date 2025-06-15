import os
import scipy.io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copia_prediccio_esquelet import load_model, predAndPrintImg
import cv2
  # Si ho poses en un fitxer separat

# Configura aquí
BASE_IMG_PATH = r'C:\Users\Asus\Desktop\TFG\SLP data\SLP\danaLab\00001\IR'
MAT_PATH = r'C:\Users\Asus\Desktop\TFG\SLP data\SLP\danaLab\00001\joints_gt_IR.mat'
CONDITIONS = ['uncover', 'cover1', 'cover2']
IMGTYPE = 'IR'

def apply_affine_transform(x, y, trans):
    """Aplica una matriu de transformació afí (2x3) als punts (x, y)."""
    pts = np.vstack([x, y, np.ones_like(x)])  
    transformed = np.dot(trans, pts)  
    return transformed[0], transformed[1]


# Carrega el GT
mat = scipy.io.loadmat(MAT_PATH)
joints_gt = mat['joints_gt']  # shape: (3, 14, N)

model = load_model(IMGTYPE)
results = {}

for cond in CONDITIONS:
    cond_path = os.path.join(BASE_IMG_PATH, cond)
    filenames = sorted([f for f in os.listdir(cond_path) if f.endswith('.png')])

    all_errors = []
    per_joint_errors = [[] for _ in range(14)]

    print(f'\n🔎 Analitzant condició: {cond} ({len(filenames)} imatges)')

    for idx, fname in enumerate(tqdm(filenames)):
        image_path = os.path.join(cond_path, fname)

        # 🔄 Obté predicció i transformació
        img_vis, pred, _, trans = predAndPrintImg(image_path, "tmp", IMGTYPE)

        if idx >= joints_gt.shape[2]:
            print(f"❗ Index {idx} fora de rang en el .mat")
            continue

        # Carrega i transforma GT
        x_gt = joints_gt[0, :, idx] 
        y_gt = joints_gt[1, :, idx] 
        v_gt = joints_gt[2, :, idx]
        
        x_gt_resized = x_gt * 1.3
        y_gt_resized = y_gt * 0.75

        x_gt_tf, y_gt_tf = apply_affine_transform(x_gt_resized, y_gt_resized, trans)
        
        scale_x = 256 / 160
        scale_y = 256 / 120

        #x_gt_tf = x_gt * scale_x
        #y_gt_tf = y_gt * scale_y
        
        print(f"\n🧪 {fname}")
        print(f"  - GT original X (min,max): {x_gt.min():.2f}, {x_gt.max():.2f}")
        print(f"  - GT original Y (min,max): {y_gt.min():.2f}, {y_gt.max():.2f}")
        print(f"  - GT transformat X (min,max): {x_gt_tf.min():.2f}, {x_gt_tf.max():.2f}")
        print(f"  - GT transformat Y (min,max): {y_gt_tf.min():.2f}, {y_gt_tf.max():.2f}")
        print(f"  - Imatge shape: {img_vis.shape}")

        

        # 🔍 Visualització
        plt.figure(figsize=(6, 6))
        plt.imshow(img_vis, cmap='gray', zorder=1)
        plt.scatter(x_gt_tf, y_gt_tf, c='lime', s=80, label='Ground Truth', zorder=5)
        plt.scatter(pred[:, 0], pred[:, 1],
            c='red', s=60, marker='x', label='Predicció', zorder=6)
        plt.title(f"{cond.upper()} - {fname}")
        plt.xlim([0, 255])
        plt.ylim([255, 0])
        plt.legend()
        #plt.show()


        # Càlcul d’errors
        for j in range(14):
            if v_gt[j] == 1:
                x_pred, y_pred = pred[j, 0], pred[j, 1]
                error = np.sqrt((x_gt_tf[j] - x_pred) ** 2 + (y_gt_tf[j] - y_pred) ** 2)
                per_joint_errors[j].append(error)
                all_errors.append(error)

    mean_per_joint = [np.mean(j) if j else 0 for j in per_joint_errors]
    mean_error = np.mean(all_errors) if all_errors else 0

    results[cond] = {
        'mean_error': mean_error,
        'per_joint_errors': mean_per_joint,
        'total_points': len(all_errors)
    }

# RESULTATS
print("\n📊 RESULTATS:")
for cond in CONDITIONS:
    print(f"\n🧾 Condició: {cond.upper()}")
    print(f"   · Error mitjà global: {results[cond]['mean_error']:.2f} px")
    for i, err in enumerate(results[cond]['per_joint_errors']):
        print(f"     - Articulació {i}: {err:.2f} px")
