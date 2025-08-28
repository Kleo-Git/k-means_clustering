## K-means_clustering for colour-based image sorting.

### Project Overview, Image Clustering and Custom Colour Gradients:
This project uses k-means clustering in order to sort images by predominant colours. This project allows visualization to observe how the clustering of images changes over iterations. Additionally, this image contains a class that can generate custom colour gradients from pre-defined colours, or use preset colour gradients. These colour gradients can be used directly for creating interesting visuals in this project, displaying clear colour regions on images.

### Background & Motivation:
Clustering is a core technique in machine learning. The main purpose of this project is to sort images by colour, but it was a valuable learning experience, giving me the chance to develop a deeper understanding of the k-means algorithm. This implementation visually interesting results while also being fairly efficient, making it suitable for experimenting with colour clustering on images of different sizes. 

### Methodology:
The **"Colour_Map_Object"** class within **colour_maps.py** manages colour maps for visualization. It includes a set of predefined colours and gradients, and provides utilities for generating smooth gradients, extracting discrete palettes (including a HSV based palette) and plotting gradients. These tools can be used to visualize image clusters consistently and clearly. This code starts with the **"Image_To_Array"** class which converts images into pixel arrays across different colours space **(RGB, LAB, HSV, XYZ, LUV)**. It supports turning 3D image arrays into flattened arrays, allowing for easy data processing when using the clustering algorithm. It also provides basic tools to visualize the given image, and the image in different colour spaces (we would expect these to be identical or near identical). Using this class to generate an array in the flattened form can then be used for the k-means clustering. This is an iterative clustering method which groups pixels together based on their colours. This implementation works by randomly intializing k number of centroids, which represent the colour of a cluster. We then assign the pixels that are closest in RGB space to each cluster, and after we assign all pixels to the clusters, we calculate the new mean RGB value of a given cluster. Then repeat the assignment step for this new mean value, and recalculate mean, repeating until there is minimal change or max iterations are reached. Lastly this project includes the images sorter class which uses results from the k-means clustering in order to find images dominant colours. It allows loading multiple images from a folder which can then split the images into warm and cool colours, or sort the images along the colour spectrum.

### Results:

First we tested to make sure the algorithm is correctly clustering colours:

