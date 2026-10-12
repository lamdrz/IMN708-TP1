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

    mask_foreground = data > 0
    
    if np.any(mask_foreground):
        data_fg = data[mask_foreground]
    else:
        data_fg = data
    stats = {
        "min": np.min(data_fg),
        "max": np.max(data_fg),
        "mean": np.mean(data_fg),
        "std": np.std(data_fg),
        "median": np.median(data_fg),
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
            "rms_contrast": rms_contrast(data[data > 0]),
            "michelson_contrast": michelson_Contrast(data[data > 0]),
            "snr": snr(data[data > 0]),
        }
    return stats

def show_stats(stats):
    print("Stats pour l'image :")
    nx, ny, nz = stats['data'].shape
    zx, zy, zz = stats['zooms']
    longueur_mm = nx * zx
    largeur_mm = ny * zy
    hauteur_mm = nz * zz

    print(f"  Longueur : {nx} voxels ({longueur_mm:.2f} mm)")
    print(f"  Largeur  : {ny} voxels ({largeur_mm:.2f} mm)")
    print(f"  Hauteur  : {nz} voxels ({hauteur_mm:.2f} mm)")
    print(f"  Dimensions (mm) : {longueur_mm:.2f} × {largeur_mm:.2f} × {hauteur_mm:.2f}")

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
    print(f"  Michelson Contrast: {stats['michelson_contrast']:.4f}")
    print(f"  SNR: {stats['snr']:.4f}")

def histogram(data, bins=100, remove_background=True, p_max=95, seuil_manuel_max=None):
    plt.figure(figsize=(10, 6))
    if remove_background:
        data_to_plot = data[data > 0].flatten() 
    else:
        data_to_plot = data.flatten()
        
    if len(data_to_plot) == 0:
        print("Erreur : Aucun voxel > 0 trouvé.")
        return
    pmin = np.percentile(data_to_plot, 1)
    
    if seuil_manuel_max is not None:
        pmax = seuil_manuel_max
    else:
        pmax = np.percentile(data_to_plot, p_max)
    data_filtered = data_to_plot[(data_to_plot >= pmin) & (data_to_plot <= pmax)]
    plt.hist(data_filtered, bins=bins, range=(pmin, pmax), color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title(f"Histogramme des tissus utiles (Plage: {pmin:.1f} à {pmax:.1f})")
    plt.xlabel("Intensité")
    plt.ylabel("Nombre de voxels")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

if __name__ == "__main__":
    args = get_args()
    stats = load_image(args.image)
    show_stats(stats)
    histogram(stats['data'][stats['data'] > 0], bins=60, remove_background=True, p_max=90)