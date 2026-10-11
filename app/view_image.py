import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse


class ImageViewer:
    def __init__(self, path: str):
        self.path = path
        self.img  = nib.as_closest_canonical(nib.load(path))
        self.data = self.img.get_fdata()
        self.nx, self.ny, self.nz = self.data.shape[:3]
        self.nt = self.data.shape[3] if self.data.ndim >= 4 else 1

        zooms = self.img.header.get_zooms()
        self.dx, self.dy, self.dz = zooms

        self.aspect_x = self.dz / self.dy
        self.aspect_y = self.dz / self.dx
        self.aspect_z = self.dy / self.dx

        self.reset()

    def reset(self):
        self.x = self.nx // 2
        self.y = self.ny // 2
        self.z = self.nz // 2
        self.t = 0

        self._setup_axes()
        self._setup_sliders()

    def _current_volume(self):
        if self.data.ndim >= 4:
            return self.data[:, :, :, self.t]
        return self.data

    def _setup_axes(self):
        plt.rcParams['axes.linewidth'] = 2
        self.fig, self.axes = plt.subplots(1, 3)

        for ax in self.axes:
            ax.set_box_aspect(1)

        volume = self._current_volume()

        self.img_x = self.axes[1].imshow(
            volume[self.x, ::-1, :].T, cmap="gray", origin="lower", aspect=self.aspect_x
        )
        self.axes[1].set_title(f"Sagittal (X = {self.x})")
        for border in self.axes[1].spines:
            self.axes[1].spines[border].set_edgecolor('green')

        self.img_y = self.axes[2].imshow(
            volume[::-1, self.y, :].T, cmap="gray", origin="lower", aspect=self.aspect_y
        )
        self.axes[2].set_title(f"Coronal (Y = {self.y})")
        for border in self.axes[2].spines:
            self.axes[2].spines[border].set_edgecolor('blue')

        self.img_z = self.axes[0].imshow(
            volume[:, :, self.z].T, cmap="gray", origin="lower", aspect=self.aspect_z
        )
        self.axes[0].set_title(f"Axial (Z = {self.z})")
        for border in self.axes[0].spines:
            self.axes[0].spines[border].set_edgecolor('red')

    def _setup_sliders(self):
        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0.28)
        slider_axes = []
        for ax in self.axes:
            pos = ax.get_position()
            slider_axes.append(
                plt.axes([pos.x0, pos.y0 - 0.12, pos.width, 0.03])
            )

        ax_z, ax_x, ax_y = slider_axes
        ax_t = plt.axes([0.15, 0.06, 0.7, 0.03])

        self.slider_x = Slider(
            ax_x, "Sagittal (X)", 0, self.nx - 1, valinit=self.x, valstep=1
        )
        self.slider_y = Slider(
            ax_y, "Coronal (Y)", 0, self.ny - 1, valinit=self.y, valstep=1
        )
        self.slider_z = Slider(
            ax_z, "Axial (Z)", 0, self.nz - 1, valinit=self.z, valstep=1
        )
        self.slider_t = Slider(
            ax_t, "Temps (T)", 0, self.nt - 1, valinit=self.t, valstep=1
        )

        for slider in (self.slider_x, self.slider_y, self.slider_z, self.slider_t):
            slider.label.set_position((0.5, -1))
            slider.label.set_horizontalalignment("center")
            slider.label.set_verticalalignment("top")

        self.slider_x.on_changed(self._on_x_change)
        self.slider_y.on_changed(self._on_y_change)
        self.slider_z.on_changed(self._on_z_change)
        self.slider_t.on_changed(self._on_t_change)

    def _on_x_change(self, val):
        self.x = int(val)
        self.img_x.set_data(self._current_volume()[self.x, ::-1, :].T)
        self.axes[1].set_title(f"Sagittal (X = {self.x})")
        self.fig.canvas.draw_idle()

    def _on_y_change(self, val):
        self.y = int(val)
        self.img_y.set_data(self._current_volume()[::-1, self.y, :].T)
        self.axes[2].set_title(f"Coronal (Y = {self.y})")
        self.fig.canvas.draw_idle()

    def _on_z_change(self, val):
        self.z = int(val)
        self.img_z.set_data(self._current_volume()[:, :, self.z].T)
        self.axes[0].set_title(f"Axial (Z = {self.z})")
        self.fig.canvas.draw_idle()

    def _on_t_change(self, val):
        self.t = int(val)
        volume = self._current_volume()
        self.img_x.set_data(volume[self.x, ::-1, :].T)
        self.img_y.set_data(volume[::-1, self.y, :].T)
        self.img_z.set_data(volume[:, :, self.z].T)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation d'une image NIfTI")
    parser.add_argument("--image", type=str, help="Chemin vers l'image NIfTI")
    args = parser.parse_args()
    if args.image:
        viewer = ImageViewer(args.image)
    else:
        image = input("NIfTI image path: ")
        viewer = ImageViewer(image)

    viewer.show()