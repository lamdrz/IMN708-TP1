from view_image import ImageViewer

viewer = ImageViewer("C:/Users/Louis/kDrive/Cours/5 - Sherbrooke/Automne/IMN708/TP/TP1/Data_TP0/IRM/Brain/flair.nii")
print(viewer.data.shape)
viewer.show()