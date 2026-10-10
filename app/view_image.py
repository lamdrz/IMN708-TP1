import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class ImageViewer:
    def __init__(self, path: str):
        self.path = path
        self.img  = nib.load(path)
        self.data = self.img.get_fdata()
        self.nx, self.ny, self.nz = self.data.shape

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

        self._setup_axes()
        self._setup_sliders()

    def _setup_axes(self):
        plt.rcParams['axes.linewidth'] = 2
        self.fig, self.axes = plt.subplots(1, 3)

        for ax in self.axes:
            ax.set_box_aspect(1)

        self.img_x = self.axes[0].imshow(
            self.data[self.x, :, :].T, cmap="gray", origin="lower", aspect=self.aspect_x
        )
        self.axes[0].set_title(f"Axial (X = {self.x})")
        for border in self.axes[0].spines:
            self.axes[0].spines[border].set_edgecolor('red')

        self.img_y = self.axes[1].imshow(
            self.data[:, self.y, :].T, cmap="gray", origin="lower", aspect=self.aspect_y
        )
        self.axes[1].set_title(f"Sagittal (Y = {self.y})")
        for border in self.axes[1].spines:
            self.axes[1].spines[border].set_edgecolor('green')

        self.img_z = self.axes[2].imshow(
            self.data[:, :, self.z].T, cmap="gray", origin="lower", aspect=self.aspect_z
        )
        self.axes[2].set_title(f"Coronal (Z = {self.z})")
        for border in self.axes[2].spines:
            self.axes[2].spines[border].set_edgecolor('blue')

    def _setup_sliders(self):
        ax_x = plt.axes([0.15, 0.14, 0.7, 0.03])
        ax_y = plt.axes([0.15, 0.09, 0.7, 0.03])
        ax_z = plt.axes([0.15, 0.04, 0.7, 0.03])

        self.slider_x = Slider(
            ax_x, "Axial (X)", 0, self.nx - 1, valinit=self.x, valstep=1
        )
        self.slider_y = Slider(
            ax_y, "Sagittal (Y)", 0, self.ny - 1, valinit=self.y, valstep=1
        )
        self.slider_z = Slider(
            ax_z, "Coronal (Z)", 0, self.nz - 1, valinit=self.z, valstep=1
        )

        self.slider_x.on_changed(self._on_x_change)
        self.slider_y.on_changed(self._on_y_change)
        self.slider_z.on_changed(self._on_z_change)

    def _on_x_change(self, val):
        self.x = int(val)
        self.img_x.set_data(self.data[self.x, :, :].T)
        self.axes[0].set_title(f"Axial (X = {self.x})")
        self.fig.canvas.draw_idle()

    def _on_y_change(self, val):
        self.y = int(val)
        self.img_y.set_data(self.data[:, self.y, :].T)
        self.axes[1].set_title(f"Sagittal (Y = {self.y})")
        self.fig.canvas.draw_idle()

    def _on_z_change(self, val):
        self.z = int(val)
        self.img_z.set_data(self.data[:, :, self.z].T)
        self.axes[2].set_title(f"Coronal (Z = {self.z})")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()