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
    parser.add_argument("axis", type=str, choices=['x', 'y', 'z'], 
                        help="Axe (x: Sagittal, y: Coronal, z: Axial)")
    parser.add_argument("type", type=str, choices=["min", "max"], 
                        help="Type de projection ('min' ou 'max').")
    parser.add_argument("--start", type=int, default=None, 
                        help="Début de la tranche (optionnel)")
    parser.add_argument("--end", type=int, default=None, 
                        help="Fin de la tranche (optionnel)")

    args = parser.parse_args()
    return args


def load_image(image_path):
    img  = nib.as_closest_canonical(nib.load(image_path))
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[:, :, :, 0]  # On gère pas la 4eme dimesion ici
    zooms = img.header.get_zooms()[:3]
    return data, zooms


def project(data, zooms, axis, type, start=None, end=None):
    ax_i = ['x', 'y', 'z'].index(axis)
    
    max_end = data.shape[ax_i]
    start = max(0, start) if start is not None else 0
    end   = min(max_end, end) if end is not None else max_end

    slice_data = data.take(indices=range(start, end), axis=ax_i)
    if type == "min":
        mip   = np.min(slice_data, axis=ax_i)
        title = 'm'
    else:
        mip   = np.max(slice_data, axis=ax_i)
        title = 'M'
    title += f"IP sur {axis.upper()} ({start} - {end})"

    dx, dy, dz = zooms
    if axis == 'x':
        aspect = dz / dy
    elif axis == 'y':
        aspect = dz / dx
    else:  # axis == 'z'
        aspect = dy / dx

    plt.imshow(mip.T, cmap="gray", origin="lower", aspect=aspect)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    args = get_args()
    data, zooms = load_image(args.image)
    project(data, zooms, args.axis, args.type, args.start, args.end)

    