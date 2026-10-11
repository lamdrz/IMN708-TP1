import argparse
from view_image import ImageViewer
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


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
    t1_brain = t1[t1 > 50]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(t1_brain.ravel(), bins=100)
    axes[0].set_title("T1")

    axes[1].hist(flair_brain.ravel(), bins=100, color='orange')
    axes[1].set_title("Flair")

    plt.show()  


if __name__ == "__main__":
    flair_path, t1_path = get_paths()
    # ImageViewer(flair_path).show()
    # ImageViewer(t1_path).show()

    flair = nib.as_closest_canonical(nib.load(flair_path)).get_fdata()
    t1 = nib.as_closest_canonical(nib.load(t1_path)).get_fdata()

    show_histograms(flair_path, t1_path)

