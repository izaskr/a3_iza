
"""
Class for plotting a heat map from a tensor in matplitlib
Requires:
- matplotlib
- numpy
Based on https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ConfusionMatrixHeatMap():
	def __init__(self, tensor, y_names, x_names, out_name):
		"""
		Params:
		tensor : numpy ndarray or torch tensor of shape n×m, where n can equal m, but not nesessarily
		y_names : list of str : names of tickmarks on y axis; should be in a desired order; the names
				will go from top to bottom along the y axis
		x_names : list of str : names of tickmarks on x axis; should be in a desired order; the names
                                will go from top to bottom along the x axis
		out_name : str : name of the saved plot
		Returns:
		None
		"""

		self.tensor = tensor
		self.y_names = y_names
		self.x_names = x_names
		self.out_name = out_name


		# convert to a numpy array and round the values to 2 decimals to be able to fit the cells
		Cnp = self.tensor.numpy()
		Cnp = np.around(Cnp, decimals=2)

		# set up the plot: size and tick marks
		fig, ax = plt.subplots(figsize=(14,14)) # in inches, ~*2 to get cm
		im = ax.imshow(self.tensor)
		ax.set_xticks(np.arange(len(x_names)))
		ax.set_yticks(np.arange(len(y_names)))

		# tick labels
		ax.set_xticklabels(x_names)
		ax.set_yticklabels(y_names)
		# tick labels: position and rotation for columns
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

		# iteratively insert the cell values into the plot; in the middle of cells and in white
		for i in range(len(y_names)):
			for j in range(len(x_names)):
				text = ax.text(j, i, Cnp[i, j], ha="center", va="center", color="w")


		# add the title to the plot
		s_title = "Heat map"
		ax.set_title(s_title)
		# add a colorbar
		plt.colorbar(im)
		fig.tight_layout()

		#Uncomment below to show the plot
		#plt.show()

		# save the plot as .png, but other formats are available (e.g. .svg or .jpg)
		plt.savefig(self.out_name+".png")
