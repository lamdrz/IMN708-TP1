from view_image import ImageViewer

viewer = ImageViewer("data/IRM/Brain/t1.nii")
print(viewer.data.shape)
viewer.show()