import argparse
from view_image import ImageViewer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
from skimage.restoration import estimate_sigma, denoise_nl_means, denoise_bilateral
from scipy.ndimage import gaussian_filter


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


def load_image(path):
    image = nib.as_closest_canonical(nib.load(path))
    data = image.get_fdata()
    if data.ndim == 4:
        data = data[:, :, :, 0]
    return data, image.header.get_zooms()[:3]


def show_histograms(flair, t1, hide_background=False):
    if hide_background:
        flair_brain = flair[flair > np.percentile(flair, 40)]
        t1_brain    = t1[t1 > np.percentile(t1, 40)]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(t1_brain.ravel(), bins=100)
    axes[0].set_title("T1")

    axes[1].hist(flair_brain.ravel(), bins=100, color='orange')
    axes[1].set_title("Flair")

    plt.show()


def sigma_region_of_interest(image, cx, cy, cz=None, half=5, half_z=None):
    if cz is None:
        roi = image[
            cx - half : cx + half, 
            cy - half : cy + half,
        ]
    else:
        roi = image[
            cx - half   : cx + half, 
            cy - half   : cy + half,
            cz - half_z : cz + half_z
        ]
    sigma = np.std(roi)
    print(f"Sigma : {sigma:.2f}")
    print(f"Sigma (Scikit-image) : {estimate_sigma(roi):.2f}")
    return sigma


def mean_roi(image, cx, cy, half=5):
    roi = image[
        cx - half : cx + half, 
        cy - half : cy + half,
    ]
    mean = np.mean(roi)
    print(f"Mean : {mean:.2f}")
    return mean

def show_roi_validation(
    image,
    spacing,
    cx,
    cy,
    cz=None,
    half=5,
    half_z=None,
    title="ROI",
    sigma=0,
    mean=None,
    sigma_half=None,
    mean_half=None,
    mean_cx=None,
    mean_cy=None,
    mean_cz=None,
):
    """
    Affiche deux régions de validation : une pour le sigma (rouge) et une pour la moyenne (bleu).
    """
    if sigma_half is None:
        sigma_half = half
    if mean_half is None:
        mean_half = half
    if mean_cx is None:
        mean_cx = cx
    if mean_cy is None:
        mean_cy = cy
    if mean_cz is None:
        mean_cz = cz

    if image.ndim == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.canvas.manager.set_window_title(f"Validation ROI - {title}")
        nx, ny = image.shape
        ax.imshow(
            np.flip(image.T, axis=1),
            cmap="gray",
            origin="lower",
            aspect=spacing[1] / spacing[0],
        )

        rect_sigma = patches.Rectangle(
            ((nx - 1) - (cx + sigma_half), cy - sigma_half),
            2 * sigma_half,
            2 * sigma_half,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
            label="sigma",
        )
        ax.add_patch(rect_sigma)

        rect_mean = patches.Rectangle(
            ((nx - 1) - (mean_cx + mean_half), mean_cy - mean_half),
            2 * mean_half,
            2 * mean_half,
            linewidth=1.5,
            edgecolor="blue",
            facecolor="none",
            linestyle="--",
            label="moyenne",
        )
        ax.add_patch(rect_mean)

        ax.set_title(f"ROI ({title})")
        ax.axis("off")
        legend = ax.legend(loc="upper right", frameon=True)
        for text in legend.get_texts():
            text.set_fontsize(10)

        metrics = f"$\\sigma_{{bruit}} = {sigma:.2f}$"
        if mean is not None:
            metrics += f" | moyenne = {mean:.2f}"
        plt.suptitle(f"{title} | {metrics}")
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title(f"Validation ROI - {title}")
    nx, ny, _ = image.shape

    coupe_axiale = np.flip(image[:, :, cz].T, axis=1)
    axes[0].imshow(coupe_axiale, cmap="gray", origin="lower", aspect=spacing[1] / spacing[0])
    rect_x_ax_sigma = (nx - 1) - (cx + sigma_half)
    rect_y_ax_sigma = cy - sigma_half
    rect_ax_sigma = patches.Rectangle(
        (rect_x_ax_sigma, rect_y_ax_sigma),
        2 * sigma_half,
        2 * sigma_half,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    axes[0].add_patch(rect_ax_sigma)

    rect_x_ax_mean = (nx - 1) - (mean_cx + mean_half)
    rect_y_ax_mean = mean_cy - mean_half
    rect_ax_mean = patches.Rectangle(
        (rect_x_ax_mean, rect_y_ax_mean),
        2 * mean_half,
        2 * mean_half,
        linewidth=1.5,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
    )
    axes[0].add_patch(rect_ax_mean)
    axes[0].set_title(f"Axial (Z={cz})")
    axes[0].axis("off")

    coupe_sagittale = np.flip(image[cx, :, :].T, axis=1)
    axes[1].imshow(coupe_sagittale, cmap="gray", origin="lower", aspect=spacing[2] / spacing[1])
    rect_x_sag_sigma = (ny - 1) - (cy + sigma_half)
    rect_y_sag_sigma = cz - half_z
    rect_sag_sigma = patches.Rectangle(
        (rect_x_sag_sigma, rect_y_sag_sigma),
        2 * sigma_half,
        2 * half_z,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    axes[1].add_patch(rect_sag_sigma)

    rect_x_sag_mean = (ny - 1) - (mean_cy + mean_half)
    rect_y_sag_mean = mean_cz - half_z
    rect_sag_mean = patches.Rectangle(
        (rect_x_sag_mean, rect_y_sag_mean),
        2 * mean_half,
        2 * half_z,
        linewidth=1.5,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
    )
    axes[1].add_patch(rect_sag_mean)
    axes[1].set_title(f"Sagittal (X={cx})")
    axes[1].axis("off")

    coupe_coronale = np.flip(image[:, cy, :].T, axis=1)
    axes[2].imshow(coupe_coronale, cmap="gray", origin="lower", aspect=spacing[2] / spacing[0])
    rect_x_cor_sigma = (nx - 1) - (cx + sigma_half)
    rect_y_cor_sigma = cz - half_z
    rect_cor_sigma = patches.Rectangle(
        (rect_x_cor_sigma, rect_y_cor_sigma),
        2 * sigma_half,
        2 * half_z,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    axes[2].add_patch(rect_cor_sigma)

    rect_x_cor_mean = (nx - 1) - (mean_cx + mean_half)
    rect_y_cor_mean = mean_cz - half_z
    rect_cor_mean = patches.Rectangle(
        (rect_x_cor_mean, rect_y_cor_mean),
        2 * mean_half,
        2 * half_z,
        linewidth=1.5,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
    )
    axes[2].add_patch(rect_cor_mean)
    axes[2].set_title(f"Coronal (Y={cy})")
    axes[2].axis("off")

    metrics = f"$\\sigma_{{bruit}} = {sigma:.2f}$"
    if mean is not None:
        metrics += f" | moyenne = {mean:.2f}"
    plt.suptitle(f"{title} | {metrics}", fontsize=14)
    plt.tight_layout()
    plt.show()


def snr(image, spacing, cx_std, cy_std, half_std, cx_mean, cy_mean, half_mean, show_roi=False, title="SNR"):
    sigma = sigma_region_of_interest(image, cx_std, cy_std, half=half_std)
    mean  = mean_roi(image, cx_mean, cy_mean, half=half_mean)
    snr   = mean / sigma
    print(f"SNR : {snr:.2f}")

    if show_roi:
        show_roi_validation(
            image, spacing,
            cx_std, cy_std, sigma_half=half_std, sigma=sigma,
            mean_cx=cx_mean, mean_cy=cy_mean, mean_half=half_mean, mean=mean,
            title=title,    
        )

    return snr


def denoise_comparison(image, sigma):
    denoised_gauss = gaussian_filter(...)
    denoised_bilat = denoise_bilateral(...)
    denoised_nlm   = denoise_nl_means(...)

    res_gauss = image - denoised_gauss
    res_bilat = image - denoised_bilat
    res_nlm   = image - denoised_nlm


if __name__ == "__main__":
    flair_path, t1_path = get_paths()
    # ImageViewer(flair_path).show()
    # ImageViewer(t1_path).show()

    flair, flair_spacing = load_image(flair_path)
    t1, t1_spacing = load_image(t1_path)

    flair_2d = flair[:, :, 11]
    t1_2d    = t1[:, :, 90]

    show_histograms(flair_2d, t1_2d, hide_background=True)

    snr_flair_2d = snr(
        flair_2d, flair_spacing,
        100, 214, 3,
        130, 115, 5,
        show_roi=True, title="Flair"
    )
    snr_t1_2d = snr(
        t1_2d, t1_spacing,
        62, 25, 2,
        75, 110, 4,
        show_roi=True, title="T1"
    )