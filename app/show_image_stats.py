import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        description="Calcule et affiche le MIP ou mIP d'une image NIfTI selon un axe."
    )
    parser.add_argument("image", type=str, 
                        help="Image NIfTI")
    
    return parser.parse_args()


def compute_raw_stats(data):
    stats = {
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "median": np.median(data),
        "shape": data.shape
    }
    return stats

def michelson_Contrast(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if (max_val + min_val) == 0:
        return 0
    return (max_val - min_val) / (max_val + min_val)

def rms_contrast(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    if mean_val == 0:
        return 0
    return std_val / mean_val

def snr(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return 0
    return mean_val / std_val

def load_image(image_path):
    img  = nib.as_closest_canonical(nib.load(image_path))
    data = img.get_fdata()
    volume_mm3 = nib.imagestats.mask_volume(img)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    zooms = img.header.get_zooms()[:3]
    stats = {
            "data": data,
            "computed_stats": compute_raw_stats(data),
            "zooms": zooms,
            "volume_mm3": volume_mm3,
            "rms_contrast": rms_contrast(data),
            "snr": snr(data),
        }
    return stats

def show_stats(stats):
    print("Stats pour l'image :")
    print(f"  Longueur: {stats['data'].shape[0]}")
    print(f"  Largeur: {stats['data'].shape[1]}")
    print(f"  Hauteur: {stats['data'].shape[2]}")
    
    zx, zy, zz = stats['zooms']
    print(f"  Zooms (Voxel): {zx:.3f} × {zy:.3f} × {zz:.3f} mm")
    
    vol_mm3 = float(stats['volume_mm3'])
    vol_cm3 = vol_mm3 / 1000
    volume_str = f"{vol_cm3:,.2f} cm³ (soit {vol_mm3:,.0f} mm³)".replace(",", " ")
    print(f"  Volume: {volume_str}")
    print(f"  Min: {stats['computed_stats']['min']:.2f}")
    print(f"  Max: {stats['computed_stats']['max']:.2f}")
    print(f"  Mean: {stats['computed_stats']['mean']:.2f}")
    print(f"  Std: {stats['computed_stats']['std']:.2f}")
    print(f"  Median: {stats['computed_stats']['median']:.2f}")
    print(f"  RMS Contrast: {stats['rms_contrast']:.4f}")
    print(f"  SNR: {stats['snr']:.4f}")

def show_matplotlib_dashboard(stats):
    data = stats['data']
    c = stats['computed_stats']
    zx, zy, zz = stats['zooms']
    vol_cm3 = float(stats['volume_mm3']) / 1000

    fig = plt.figure(figsize=(15, 5))
    fig.canvas.manager.set_window_title('Dashboard IRM')
    fig.suptitle("Tableau de bord de l'image NIfTI", fontsize=16, fontweight='bold')

    # --- 1. Affichage de la coupe centrale (Vue axiale) ---
    ax1 = fig.add_subplot(1, 3, 1)
    mid_z = data.shape[2] // 2
    # Transposition pour un affichage souvent plus naturel des NIfTI
    ax1.imshow(data[:, :, mid_z].T, cmap='gray', origin='lower')
    ax1.set_title(f'Coupe centrale (Z = {mid_z})')
    ax1.axis('off')

    # --- 2. Histogramme des intensités ---
    ax2 = fig.add_subplot(1, 3, 2)
    # On ignore souvent les voxels à 0 (le fond) pour mieux voir les tissus
    voxels_actifs = data[data > 0.01].flatten() 
    ax2.hist(voxels_actifs, bins=64, color='steelblue', alpha=0.8)
    ax2.set_title("Histogramme des intensités (> 0)")
    ax2.set_xlabel("Niveaux de gris")
    ax2.set_ylabel("Nombre de voxels")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # --- 3. Affichage des Statistiques ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off') # On cache les axes pour ne garder que le texte
    
    # Préparation du texte formaté
    texte = (
        f"GÉOMÉTRIE\n"
        f"---------------------------\n"
        f"Dimensions : {c['shape'][0]} × {c['shape'][1]} × {c['shape'][2]}\n"
        f"Voxel (mm) : {zx:.3f} × {zy:.3f} × {zz:.3f}\n"
        f"Volume     : {vol_cm3:,.2f} cm³\n\n"
        f"INTENSITÉS\n"
        f"---------------------------\n"
        f"Min / Max  : {c['min']:.1f} / {c['max']:.1f}\n"
        f"Moyenne    : {c['mean']:.1f}\n"
        f"Médiane    : {c['median']:.1f}\n"
        f"Écart-type : {c['std']:.1f}\n\n"
        f"QUALITÉ\n"
        f"---------------------------\n"
        f"Contraste RMS : {stats['rms_contrast']:.4f}\n"
        f"SNR           : {stats['snr']:.4f}"
    ).replace(',', ' ') # Remplacement pour les espaces des milliers

    # Ajout du bloc de texte dans le 3ème cadran
    ax3.text(0.1, 0.95, texte, fontsize=11, family='monospace', 
             va='top', ha='left', transform=ax3.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # Ajustement des marges et affichage
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    stats = load_image(args.image)
    show_matplotlib_dashboard(stats)