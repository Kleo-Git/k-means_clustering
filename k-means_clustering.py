from image_to_array import Image_To_ArrayRGB
import colorsys
import numpy as np
import random as rd
import cv2
import matplotlib.pyplot as plt

class KMeans_Image:
    
    def __init__(self, image_object, k):
        """
        Intialize KMeans_Image object

        Parameters
        ----------
        input_array : list of list[int]
            The list where each element is [R,G,B] for a given pixel.
            The shape of the list will be (height*width, 3).
        k : int
            Number of centroids used for k-means clustering.

        Returns
        -------
        None.

        """
        self.image_object = image_object
        self.input_array = image_object.rgb_array
        self.image = image_object.image
        self.image.shape = image_object.image_shape
        self.k = k
        
        #Create inital centroid values
        self.centroids = self.initalize_centroids()
        
        self.colours = []
        
    def calculate_distance(position1, position2):
        """
        Compute the Eucledian distance between 2 points in n-dimensional space.

        Parameters
        ----------
        position1 : list
            Coordinates of the first point.
        position2 : list
            Coordinates of the second point.

        Returns
        -------
        distance : float
            The total distance between position1 and position2.

        """
        #Calculates the squared sum of the differences in x,y,z positions.
        squared_sum = sum((a-b) ** 2 for a, b in zip(position1,position2))
        
        #Square root the squared sum to get total distance using pythagoras' theorem.
        distance = np.sqrt(squared_sum)
        
        #Return distance
        return distance
    
    def initalize_centroids(self):
        """
        Randomly selects k unique centroids from the flattened input array

        Returns
        -------
        picked_values : list
            List of indices of randomly selected centroids.
            Each index corresponds to a pixel in self.input_array (flattened row-wise).

        """
        
        #Total amount of pixels will equal length of array, given current setup
        num_pixels = len(self.input_array)
        
        #Store a list of values that have already been chosen as a centroid
        #Prevents picking the same pixel for multiple centroids
        picked_values = []
        
        #If k is larger then number of pixels would cause an infinite loop
        if self.k > num_pixels:
            print("Attempting to pick too many centroids for given image size")
            return
        
        #Should be much more efficient to use this.
        #random.sample(range(num_pixels), k)
        
        #Use a counter so it continually picks centroids until num centroids = k
        #This guarantees we reach k centroids, despite random picks and preventing duplicates
        counter = 0
        while counter < self.k:
            #Pick a random pixel from the list of pixels
            pixel = rd.randint(0,num_pixels-1)
            
            #Prevent allowing for repeated pixels for a given centroid
            if pixel not in picked_values:
                picked_values.append(pixel)
                counter+=1
                
        #Return the selected centroids, list that should have length k
        return picked_values
            
    def k_means(self):
        
        print("Centroids = ", self.centroids)
        
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        
        #Empty list that will be in the form of [[R,G,B], "Associated Centroid"]
        new_list = []
        
        for i in range(len(self.input_array)):
            shortest_distance = float("inf")
            associated_centroid = 0
            
            for centroid in self.centroids:
                
                current_distance = self.calculate_distance(self.input_array[i], self.input_array[centroid])
                
                if current_distance < shortest_distance:
                    shortest_distance = current_distance
                    associated_centroid = centroid
                
            new_list.append([self.input_array[i], associated_centroid])
            
        self.colour_centroids(a.image, self.centroids)
        
        return new_list
    
    def colour_centroids(self, saturation = 0.8, value = 0.9):
        """
        Colour the centroid pixels of the image for visualization.
        
        Each centroid in self.centroids is assinged a distinct colour
        generated from HSV colour space. The colours are displayed on a 
        copy of the image to avoid modifying values of the original.

        Parameters
        ----------
        saturation : float, optional
            Saturation of the HSV colour (0 to 1). Higher values produce more vivid colours.
            Default is 0.8.
        value : float, optional
            Value/brightness of the HSV colour (0 to 1). Higher values produce brighter colours.
            The default is 0.9.

        Returns
        -------
        None.
            Displayes the image with centroid pixels coloured using matplotlib.

        """
        
        #Generate distinct colours for each centroid.
        colours = self.colour_map(saturation, value)
        
        #Make a copy of image so original is not modified.
        temp_image = self.image.copy()
        
        for i, centroid in enumerate(self.centroids):
            #Find corresponding pixel positions for recolouring.
            x = (centroid) // self.image.shape[0]
            y = (centroid) % self.image.shape[1]
            
            #Assign the colour to the centroid pixel.
            temp_image[x][y] = colours[i]
        
        #Display recoloured image.
        plt.imshow(temp_image)
        plt.show()
        
    def colour_map(self, saturation, value):
        """
        Generates a list of distinct RGB colours.
        
        Colours are created by evenly spacing hues around the HSV colour wheel
        while using the specified saturation and value. The HSV values are converted
        to RGB and scaled to 0-255.
        

        Parameters
        ----------
        saturation : float
            Saturation of the HSV colour (0 to 1). Higher values produce more vivid colours.
        value : float
            Value/brightness of the HSV colour (0 to 1). Higher values produce brighter colours.

        Returns
        -------
        colours_corrected : list of list[int]
            List of RGB colours for each centroid, where each colour is [R,G,B] with 
            values scaled to 0-255 as intergers.

        """
    
        colours=[]
        
        #Generates HSV colours evenly spaced by hue.
        for i in range(self.k):
            hue = i / self.k
            colours.append(list(colorsys.hsv_to_rgb(hue, saturation, value)))
    
        colours_corrected = []
        
        #Convert to 0-255 RGB intergers
        for i in range(len(colours)):
            colours_corrected.append([int(j * 255) for j in colours[i]])
            
        return colours_corrected

a = Image_To_ArrayRGB("images/test_image4.jpg")
b = KMeans_Image(a, 3)
b.colour_centroids()





