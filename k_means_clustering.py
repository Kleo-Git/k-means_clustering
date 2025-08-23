from image_to_array import Image_To_Array
from colour_maps import Colour_Map_Object
from skimage import color
import cv2 as cv
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time

class KMeans_Image:
    
    def __init__(self, image_object, k=None, colours=None, colour_space=None, ignore_background = False, background_threshold = 10, messages=False):
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
        
        self.original_input_array = image_object.colour_array
        self.original_rgb_array = image_object.rgb_array
        
        self.image = image_object.image
        self.image_shape = image_object.image_shape
        
        self.k = k
        if self.k == None:
            self.k = self.evaluate_k()
        
        self.colour_space = image_object.colour_space
        
        self.ignore_background = ignore_background
        self.background_threshold = background_threshold
        
        self.messages = messages
        
        if self.ignore_background:
            self.__setup_filtered_arrays()
            if self.messages:
                print(f"Using {len(self.input_array)} foreground pixels out of {len(self.original_input_array)} total")
        else:
            self.input_array = self.original_input_array
            self.rgb_array = self.original_rgb_array
            self.pixel_to_original_map = list(range(len(self.input_array)))
        
        self.final_centroids = None
        self.final_assignments = None
        
        #Colours used for visualization
        self.colours = colours
        if self.colours == None:
            self.colours = Colour_Map_Object().hsv_colour_map(self.k)
            
    def __setup_filtered_arrays(self):
        rgb_array = np.array(self.original_rgb_array)
        
        black_mask = np.all(rgb_array <= self.background_threshold, axis=1)
        
        #white_mask = np.all(rgb_array >= (255-self.background_threshold), axis=1)
        
        background_mask = black_mask # | white_mask
        
        foreground_indicies = np.where(~background_mask)[0]
        
        self.input_array = [self.original_input_array[i] for i in foreground_indicies]
        self.rgb_array = [self.original_rgb_array[i] for i in foreground_indicies]

        self.pixel_to_original_map = foreground_indicies.tolist()
        
    def __convert_colour_space(self, rgb_array):
        """
        Convert RGB array to specified colour space
        """
        rgb_normalized = np.array(rgb_array) / 255
        
        if self.colour_space == "RGB":
            return rgb_normalized*255
        
        elif self.colour_space == "LAB":
            lab = color.rgb2lab(rgb_normalized.reshape(-1,1,3)).reshape(-1,3)
            return lab
            
        elif self.colour_space == "HSV":
            hsv = color.rgb2hsv(rgb_normalized.reshape(-1,1,3)).reshape(-1,3)
            hsv[:, 0] *= 360
            hsv[:, 1:] *= 100
            return hsv
        
        elif self.colour_space == "XYZ":
            xyz = color.rgb2xyz(rgb_normalized.reshape(-1,1,3)).reshape(-1,3)
            return xyz
        
        elif self.colour_space == "LUV":
            luv = color.rgb2luv(rgb_normalized.reshape(-1,1,3)).reshape(-1,3)
            return luv
        
        else:
            raise ValueError(f"Unrecognised colour space : {self.colour_space}")
            
    def __convert_back_to_rgb(self, colour_values):
        """
        Convert color values back to RGB for display
        
        Parameters
        ----------
        color_values : numpy array
            Values in the current color space
            
        Returns
        -------
        rgb_values : numpy array
            RGB values (0-255)
        """
        
        colour_values = np.array(colour_values)
        
        if self.colour_space == "RGB":
            return np.clip(colour_values, 0, 255).astype(int)
        
        elif self.colour_space == "LAB":
            # Convert LAB back to RGB
            lab_normalized = colour_values.reshape(-1, 1, 3)
            rgb = color.lab2rgb(lab_normalized).reshape(-1, 3)
            return np.clip(rgb * 255, 0, 255).astype(int)
        
        elif self.colour_space == "HSV":
            # Convert HSV back to RGB using OpenCV format
            hsv_img = colour_values.reshape(1, -1, 3).astype(np.uint8)
            rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)
            return rgb_img.reshape(-1, 3)
        
        elif self.colour_space == "XYZ":
            # Convert XYZ back to RGB
            xyz_normalized = (colour_values / 100).reshape(-1, 1, 3)
            rgb = color.xyz2rgb(xyz_normalized).reshape(-1, 3)
            return np.clip(rgb * 255, 0, 255).astype(int)
        
        elif self.colour_space == "LUV":
            # Convert LUV back to RGB
            luv_normalized = colour_values.reshape(-1, 1, 3)
            rgb = color.luv2rgb(luv_normalized).reshape(-1, 3)
            return np.clip(rgb * 255, 0, 255).astype(int)
        
        else:
            raise ValueError(f"Unsupported color space: {self.colour_space}")
    
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

        #Convert to NumPy arrays
        pixels = np.asarray(self.input_array)
        centroids = np.asarray(centroids)
    
        # Compute distances using NumPy
        distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
    
        # Pick nearest centroid index for each pixel
        nearest = np.argmin(distances, axis=1)
    
        #Recreate original structure of assignments
        assignments = [[pixel.tolist(), idx] for pixel, idx in zip(pixels, nearest)]
    
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
                new_centroids.append(group.mean(axis=0).tolist())

        return new_centroids
            
    def kmeans_loop(self, max_iterations = 10, tolerance=1e-3, display_interval = 0):
        """
        Perform K-means clustering algortihm on the input array.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations to run.
            Default is 10.
        tolerance : float, optional
            Threshold for minimum centroid movement to determine convergence.
            Default is 1e-3.
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
        if self.messages:
            print(f"Starting k-means clustering in {self.colour_space} color space:")
        
        #Initialize centroids
        centroids, centroid_indices = self.__initalize_centroids()
        
        for i in range(max_iterations):
            #Assign each pixel to nearest centroid
            assignments = self.assign_to_centroids(centroids)
            
            #Compute new centroids
            new_centroids = self.update_centroids(assignments, centroids)
            
            #Display the clustering progress
            if display_interval and i % display_interval == 0:
                self.visualize_clustering(assignments, centroids)
            
            #Check if centroids have converged
            if np.allclose(new_centroids, centroids, atol=tolerance):
                if self.messages:
                    print(f"Convergence reach after {i} iterations")
                break
            
            centroids = new_centroids
        else:
            print(f"Reached maximum iterations ({max_iterations})")
        
        self.final_centroids = centroids
        self.final_assignments = assignments
        
        self.wcss = self.calculate_wcss()
        
        return centroids, assignments, centroid_indices
    
    def calculate_wcss(self):
        """
        Calculates within-cluster sum of squares.
        """
        if self.final_centroids is None or self.final_assignments is None:
            return None
    
        # Separate pixels and their assignments into NumPy arrays
        # This is the corrected section.
        assignments = [item[1] for item in self.final_assignments]
        pixels = [item[0] for item in self.final_assignments]
        
        final_assignments_np = np.array(assignments)
        pixels_np = np.array(pixels)
        final_centroids_np = np.array(self.final_centroids)
        
        wcss = 0
        
        # Loop through each cluster
        for k in range(len(final_centroids_np)):
            # Get pixels assigned to this cluster using boolean indexing
            cluster_pixels = pixels_np[final_assignments_np == k]
            
            if len(cluster_pixels) > 0:
                # Get the centroid for this cluster
                centroid = final_centroids_np[k]
                
                # Calculate the sum of squared distances for this cluster
                wcss += np.sum(np.linalg.norm(cluster_pixels - centroid, axis=1)**2)
                
        return wcss
        
    def visualize_clustering(self, assignments, current_centroids):
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
        rgb_centroids = self.__convert_back_to_rgb(np.array(current_centroids))
        
        #Make a copy of image so original is not modified.
        #temp_image = self.image.copy().reshape(-1, 3)
        temp_image = np.zeros((len(self.original_input_array), 3), dtype=np.uint8)
        
        if self.ignore_background:
            temp_image[:] = [0,0,0]
            
            for i, (_, centroid_idx) in enumerate(assignments):
                original_pixel_idx = self.pixel_to_original_map[i]
                temp_image[original_pixel_idx] = rgb_centroids[centroid_idx]
        else:
            for i, (_, centroid_idx) in enumerate(assignments):
                temp_image[i] = rgb_centroids[centroid_idx]
           
        temp_image = temp_image.reshape(self.image.shape)
        #Display recoloured image.
        #plt.imshow(temp_image)
        #plt.show()
        
        # Display the image
        plt.figure(figsize=(10, 5))
        
        # Show original
        plt.subplot(1, 2, 1)
        plt.imshow(self.image_object.rgb_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Show clustered
        plt.subplot(1, 2, 2)
        plt.imshow(temp_image)
        plt.title(f'K-Means Clustering (k={self.k}, {self.colour_space})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_with_gradient_colors(self):
        """
        Visualize the final clustering result using the custom gradient colors.
        
        Returns
        -------
        None.
            Displayes the image with centroid pixels coloured using matplotlib.

        
        """
        if self.final_centroids is None or self.final_assignments is None:
            print("No clustering results available. Run kmeans_loop() first.")
            return
        
        
        # Create output image using gradient colors
        if self.ignore_background:

            temp_image = np.zeros((len(self.original_input_array), 3), dtype=np.uint8)
            temp_image[:] = [0,0,0]
            
            for i, (_, centroid_idx) in enumerate(self.final_assignments):
                original_pixel_idx = self.pixel_to_original_map[i]
                temp_image[original_pixel_idx] = self.colours[centroid_idx]
                
            # Reshape back to original image dimensions
            temp_image = temp_image.reshape(self.image_shape)
            
        else:
            
            temp_image = np.zeros((len(self.final_assignments), 3), dtype=np.uint8)
            
            # Assign each pixel the gradient color of its centroid
            for i, (_, centroid_idx) in enumerate(self.final_assignments):
                temp_image[i] = self.colours[centroid_idx]
                
            temp_image = temp_image.reshape(self.image_shape)
        
        # Display comparison
        plt.figure(figsize=(10, 5))
        
        # Show original
        plt.subplot(1, 3, 1)
        plt.imshow(self.image_object.rgb_image)
        plt.title('Original Image')
        plt.axis('off')

        # Show clustered with gradient colors
        plt.subplot(1, 3, 2)
        plt.imshow(temp_image)
        plt.title('Custom Gradient Colors')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dominant_colors(self):
        """
        Get the dominant colors from clustering results
        
        Returns
        -------
        colors_rgb : list
            List of dominant colors in RGB format
        percentages : list  
            Percentage of image each color represents
        """
        if self.final_centroids is None or self.final_assignments is None:
            print("No clustering results available. Run kmeans_loop() first.")
            return None, None
        
        # Count pixels per cluster
        cluster_counts = {}
        for _, centroid_idx in self.final_assignments:
            cluster_counts[centroid_idx] = cluster_counts.get(centroid_idx, 0) + 1
        
        # Convert centroids to RGB
        colors_rgb = self.__convert_back_to_rgb(self.final_centroids)
        
        # Calculate percentages
        total_pixels = len(self.final_assignments)
        percentages = [(cluster_counts.get(i, 0) / total_pixels) * 100 
                      for i in range(self.k)]
        
        return colors_rgb.tolist(), percentages
    
    def evaluate_k(self, c_1=1, c_2=1):
        
        rgb_array_np = np.array(self.rgb_array)
        
        num_pixels = len(rgb_array_np)
        
        r_var= np.var(rgb_array_np[:, 0])
        g_var = np.var(rgb_array_np[:, 1])
        b_var= np.var(rgb_array_np[:, 2])
        
        #Maximum variance = half pixels at 0, half pixels at 255.
        #Equvialently, a deviation of 127.5 for all pixels from the mean.
        #Deviation = 127.5**2 = 16256.25
        #Colour variance shoudl be normalized by 3*16256.25 = 48768.75
        #Assuming a max image size of roughly 3840x2160 (4k image)
        #Will choose a normailization value of 10million.
        
        colour_variance = r_var + g_var + b_var
        #print("Colour Variance = ", colour_variance)
        
        colour_variance /= 48768.75
        num_pixels_corrected = num_pixels / 1e7
        
        #print("Number of pixels = ", num_pixels)
        
        
        combined_score = colour_variance * num_pixels_corrected
        
        #print(combined_score)
        
        estimated_k = int(c_1 * np.log(1+combined_score) + c_2)
        
        return estimated_k
        
        
    
    def compare_color_spaces(self, color_spaces=["RGB", "LAB", "HSV", "XYZ", "LUV"]):
        """
        Compare clustering results across different color spaces
        
        Parameters
        ----------
        color_spaces : list
            List of color spaces to compare
        """
        fig, axes = plt.subplots(1, len(color_spaces) + 1, figsize=(4 * (len(color_spaces) + 1), 4))
        
        # Show original
        axes[0].imshow(self.image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        for i, space in enumerate(color_spaces):
            # Create new instance with different color space
            kmeans_temp = KMeans_Image(self.image_object, self.k, space)
            centroids, assignments, _ = kmeans_temp.kmeans_loop(max_iterations=10)
            
            # Visualize
            display_centroids = kmeans_temp.__convert_back_to_rgb(centroids)
            clustered_image = np.zeros_like(self.rgb_array)
            for j, (_, centroid_idx) in enumerate(assignments):
                clustered_image[j] = display_centroids[centroid_idx]
            
            clustered_image = clustered_image.reshape(self.image_shape)
            axes[i + 1].imshow(clustered_image.astype(int))
            axes[i + 1].set_title(f"{space} (k={self.k})")
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()

    
start = time.time()
if __name__ == "__main__":
    k=6
    image_rgb = Image_To_Array("images/sunset2.jpg", "RGB")
    #image_rgb.show_image()
    #colour_object = Colour_Map_Object()
    #colours=colour_object.gradient_colour_map(k, "void")
    kmeans_rgb = KMeans_Image(image_rgb, k, ignore_background=False)
    kmeans_rgb.kmeans_loop(tolerance=1e-3,max_iterations=10, display_interval=1)
    #kmeans_rgb.visualize_with_gradient_colors()
    #print(kmeans_rgb.wcss)
    end=time.time()
    print(end-start)