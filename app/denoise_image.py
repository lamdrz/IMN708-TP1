import argparse
# from view_image import ImageViewer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
from skimage.restoration import estimate_sigma


def get_paths():
    parser = argparse.ArgumentParser(description="Visualisation d'images NIfTI.")
    parser.add_argument("--flair", type=str, help="Flair : ")
    parser.add_argument("--t1", type=str, help="T1 : ")
    args = parser.parse_args()

    flair, t1 = args.flair, args.t1
    if not args.flair:
        flair = input("Flair : ")
    if not args.t1:
        t1 = input("T1 : ")

    return flair, t1


def show_histograms(flair_path, t1_path):
    # Ne prend pas en compte le fond
    flair_brain = flair[flair > 50]
    t1_brain    = t1[t1 > 50]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(t1_brain.ravel(), bins=100)
    axes[0].set_title("T1")

    axes[1].hist(flair_brain.ravel(), bins=100, color='orange')
    axes[1].set_title("Flair")

    plt.show()


def sigma_region_of_interest(image, cx, cy, cz, half, half_z):
    roi = image[
        cx - half   : cx + half, 
        cy - half   : cy + half,
        cz - half_z : cz + half_z
    ]
    sigma = np.std(roi)
    print(f"Sigma : {sigma:.2f}")
    print(f"Sigma (Scikit-image) : {estimate_sigma(roi):.2f}")
    return sigma


def show_roi_validation(image, spacing, cx, cy, cz, half, half_z, title, sigma):
    """
    Fonction générée par IA pour vérifier visuellement la région d'intérêt utilisée pour calculer le sigma du bruit."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title(f"Validation ROI - {title}")
    nx, ny, _ = image.shape

    coupe_axiale = np.flip(image[:, :, cz].T, axis=1)
    axes[0].imshow(coupe_axiale, cmap="gray", origin="lower", aspect=spacing[1] / spacing[0])
    rect_x_ax = (nx - 1) - (cx + half)
    rect_y_ax = cy - half
    rect_ax = patches.Rectangle((rect_x_ax, rect_y_ax), 2 * half, 2 * half, 
                                linewidth=1.5, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect_ax)
    axes[0].set_title(f"Axial (Z={cz})")
    axes[0].axis("off")

    coupe_sagittale = np.flip(image[cx, :, :].T, axis=1)
    axes[1].imshow(coupe_sagittale, cmap="gray", origin="lower", aspect=spacing[2] / spacing[1])
    rect_x_sag = (ny - 1) - (cy + half)
    rect_y_sag = cz - half_z
    rect_sag = patches.Rectangle((rect_x_sag, rect_y_sag), 2 * half, 2 * half_z, 
                                 linewidth=1.5, edgecolor="red", facecolor="none")
    axes[1].add_patch(rect_sag)
    axes[1].set_title(f"Sagittal (X={cx})")
    axes[1].axis("off")

    coupe_coronale = np.flip(image[:, cy, :].T, axis=1)
    axes[2].imshow(coupe_coronale, cmap="gray", origin="lower", aspect=spacing[2] / spacing[0])
    rect_x_cor = (nx - 1) - (cx + half)
    rect_y_cor = cz - half_z
    rect_cor = patches.Rectangle((rect_x_cor, rect_y_cor), 2 * half, 2 * half_z, 
                                 linewidth=1.5, edgecolor="red", facecolor="none")
    axes[2].add_patch(rect_cor)
    axes[2].set_title(f"Coronal (Y={cy})")
    axes[2].axis("off")

    plt.suptitle(f"{title} | $\sigma_{{bruit}} = {sigma:.2f}$", fontsize=14)
    plt.tight_layout()
    plt.show()


def load_image(path):
    image = nib.as_closest_canonical(nib.load(path))
    data = image.get_fdata()
    if data.ndim == 4:
        data = data[:, :, :, 0]
    return data, image.header.get_zooms()[:3]


if __name__ == "__main__":
    flair_path, t1_path = get_paths()
    # ImageViewer(flair_path).show()
    # ImageViewer(t1_path).show()

    flair, flair_spacing = load_image(flair_path)
    t1, t1_spacing = load_image(t1_path)

    show_histograms(flair, t1)

    sigma_flair = sigma_region_of_interest(flair, 145, 145, 10, 5, 1)
    sigma_t1    = sigma_region_of_interest(t1, 105, 105, 100, 5, 2)

    show_roi_validation(flair, flair_spacing, 145, 145, 10, 5, 1, "Flair", sigma_flair)
    show_roi_validation(t1, t1_spacing, 105, 105, 100, 5, 2, "T1", sigma_t1)
