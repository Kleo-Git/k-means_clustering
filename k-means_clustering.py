from image_to_array import Image_To_Array
from colour_maps import Colour_Map_Object
import colorsys
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class KMeans_Image:
    
    def __init__(self, image_object, k, colours=None):
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
        centroids, indicies = self.__initalize_centroids()
        self.initial_centroids = centroids
        self.initial_centroids_indicies = indicies
        
        self.colours = colours
        
        if self.colours == None:
            self.colours = Colour_Map_Object().hsv_colour_map(self.k)
        
    def calculate_distance(self, position1, position2):
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
    
    def __initalize_centroids(self):
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
        #Picked indicies will be used for visualisation
        picked_values = []
        picked_indices = []
        
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
            if pixel not in picked_indices:
                picked_indices.append(pixel)
                picked_values.append(self.input_array[pixel])
                counter+=1
                
        #Return the selected centroids, and their indicies for colouring the image
        return picked_values, picked_indices
            
    def assign_to_centroids(self, centroids):
        """
        Assign each pixel in the input array to the nearest centroid based on 
        Euclidean distance in RGB space.

        Parameters
        ----------
        centroids : list of list[float]
            List of current centroid coordinates, where each centroid is [R,G,B].

        Returns
        -------
        assignments : list
            List of [pixel, assigned_centroid_index] pairs, where "pixel" is the original
            pixel value and "assigned_centroid_index" is the index of the nearest centroid.

        """

        #Empty list that will be in the form of [[R,G,B], "Associated Centroid"]
        assignments = []
        
        #Iterate through every pixel in the input array
        for pixel in self.input_array:
            shortest_distance = float("inf")
            associated_centroid = 0
            
            #Compare the pixel to each centroid
            for i, centroid in enumerate(centroids):

                current_distance = self.calculate_distance(pixel, centroid)

                #Update the nearest centroid if the new distance is smaller                
                if current_distance < shortest_distance:
                    shortest_distance = current_distance
                    associated_centroid = i
            
            #Record the pixel and its nearest centroid
            assignments.append([pixel, associated_centroid])
        
        return assignments
    
    def update_centroids(self, assignment_list, current_centroids):
        """
        Compute the new centroid positions as the mean of all pixels
        assigned to the given centroid.

        Parameters
        ----------
        assignment_list : list
            List of [pixel, assigned_centroid_index] pairs from "assign_to_centroids".
        current_centroids : list of list[float]
            Current centroid coordinates.

        Returns
        -------
        new_centroids : list of list[float]
            Updated centroid coordinates based on mean of assigned pixels.
            If a centroid has no assigned pixels, it retains its previous position.

        """
        #Create dictionary to group pixels by their assigned centroid index
        centroid_groups = {i: [] for i in range(self.k)}
    
        #Create the groups
        for pixel, assigned_index in assignment_list:
            centroid_groups[assigned_index].append(pixel)
        
        #Compute new centroids
        new_centroids = []
        for i in range(self.k):
            group = np.array(centroid_groups[i])
            if len(group) == 0:
                #If no pixels assigned, keep current centroid
                new_centroids.append(current_centroids[i])
            else:
                #Compute mean across all assigned pixels
                new_centroids.append(group.mean(axis=0))

        return new_centroids
            
    def kmeans_loop(self, max_iterations = 100, tolerance=1e-4, display_interval = 1):
        """
        Perform K-means clustering algortihm on the input array.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations to run.
            Default is 100.
        tolerance : float, optional
            Threshold for minimum centroid movement to determine convergence.
            Default is 1e-4.
        display_interval : int, optional
            Number of iterations between visual outputs of centriod assignments.
            Default is 1.

        Returns
        -------
        centroids : list of list[float]
            Final centroid positions after convergence (or max iterations).
        assignments : list
            Final assignments of each pixel to a centroid.
        centroid_indices : list
            Initial indicies of centroids.
        """
        #Initialize centroids
        centroids, centroid_indices = self.__initalize_centroids()
        
        for i in range(max_iterations):
            #Assign each pixel to nearest centroid
            assignments = self.assign_to_centroids(centroids)
            
            #Compute new centroids
            new_centroids = self.update_centroids(assignments, centroids)
            
            #Display the clustering progress
            if i % display_interval == 0:
                self.colour_centroids(assignments)
            
            #Check if centroids have converged
            if np.allclose(new_centroids, centroids, atol=tolerance):
                print(f"Convergence reach after {i} iterations")
                break
            
            centroids = new_centroids
        
        return centroids, assignments, centroid_indices
    
    def colour_centroids(self, assignments):
        """
        Colour the centroid pixels of the image for visualization.
        
        Each centroid in self.centroids is assinged a distinct colour
        generated from HSV colour space by default. The colours are displayed on a 
        copy of the image to avoid modifying values of the original.

        Returns
        -------
        None.
            Displayes the image with centroid pixels coloured using matplotlib.

        """
        
        #Generate distinct colours for each centroid.
        #colours = self.colour_map(saturation, value)
        
        #Make a copy of image so original is not modified.
        temp_image = self.image.copy().reshape(-1, 3)
        
        for i, (_, centroid_idx) in enumerate(assignments):
            temp_image[i] = self.colours[centroid_idx]
           
        temp_image = temp_image.reshape(self.image.shape)
        #Display recoloured image.
        plt.imshow(temp_image)
        plt.show()
        
    

a = Image_To_Array("images/lion_test.jpg")
colour_object = Colour_Map_Object()
k=6
colours=colour_object.gradient_colour_map(k, "ply1")
b = KMeans_Image(a, k, colours)  
b.kmeans_loop(max_iterations=100)
a.show_image()

