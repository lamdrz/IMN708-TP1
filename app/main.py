from view_image import ImageViewer

viewer = ImageViewer("data/IRM/Brain/flair.nii")
print(viewer.data.shape)
viewer.show()