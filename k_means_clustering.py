from image_to_array import Image_To_Array
from colour_maps import Colour_Map_Object
from skimage import color
import cv2 as cv
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time

class KMeans_Image:
    
    def __init__(self, image_object, k, colours=None, colour_space=None, ignore_background = False, background_threshold = 5, messages=False):
        """
        Intialize KMeans_Image object for k-means clustering on image data.

        Parameters
        ----------
        image_object : object
            Image object containing image specific data attributes. Colour array
            should be [H*W, 3] with each row representing [R,G,B] values for a pixel.
        k : int
            Number of centroids used for k-means clustering.
        colours : array[R,G,B], optional
            Custom colors to use for visualization. Default is None.
            default = None.
        ignore_background : bool, optional
            Whether to ignore background during clustering. Primarily useful for pixel
            art images / images with empty background. 
            default = False
        background_threshold : int, optional
            Threshold value for determining background pixels when ignore_background
            is True. Pixels with all RGB values <= this threshold are considered background.
            default = 5
        messages : bool, optional
            Whether to display certain messages during processing.
            default = False
        
        Attributes
        ----------
        image_object : object
            Reference to the input image object
        original_input_array : array
            Copy of the original colour array from image_object  
        original_rgb_array : array
            Copy of the original RGB array from image_object
        image : array-like
            The image data from image_object
        image_shape : tuple
            Shape of the image from image_object
        k : int
            Number of clusters
        colour_space : str
            Color space of the image from image_object
        ignore_background : bool
            Background ignoring flag
        background_threshold : int
            Background threshold value
        messages : bool
            Message display flag
        
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
        
        self.colour_space = image_object.colour_space
        
        self.ignore_background = ignore_background
        self.background_threshold = background_threshold
        
        self.messages = messages
        
        if self.ignore_background:
            #Ignores pixels within the threshold
            self.__setup_filtered_arrays()
            if self.messages:
                print(f"Using {len(self.input_array)} foreground pixels out of {len(self.original_input_array)} total")
        else:
            self.input_array = self.original_input_array
            self.rgb_array = self.original_rgb_array
            self.pixel_to_original_map = list(range(len(self.input_array)))
        
        #Used for evaluating end results, and visualizing with gradient
        self.final_centroids = None
        self.final_assignments = None
        
        #Colours used for visualization
        self.colours = colours
        if self.colours == None:
            self.colours = Colour_Map_Object().hsv_colour_map(self.k)
            
    def __setup_filtered_arrays(self):
        """
        Filter out background pixels from the input arrays based on background_threshold
        
        Creates filtered versions of the input_array and rgb_array containing only
        foreground pixels. Background pixels are identified as those with all RGB
        values <= background_threshold.

        Returns
        -------
        None.

        """
        
        #Convert to NumPy array
        rgb_array = np.array(self.original_rgb_array)
        
        #Find all values that are considered 'black' pixels to ignore
        black_mask = np.all(rgb_array <= self.background_threshold, axis=1)
        
        #Unused since background defaults to black and can invalidate images with
        #white pixels.
        #white_mask = np.all(rgb_array >= (255-self.background_threshold), axis=1)
        
        #Currently only black background is considered
        background_mask = black_mask # | white_mask
        
        #Get indicies of pixels that are not background
        foreground_indicies = np.where(~background_mask)[0]
        
        #Filter input and RGB arrays to only include foreground pixels
        self.input_array = [self.original_input_array[i] for i in foreground_indicies]
        self.rgb_array = [self.original_rgb_array[i] for i in foreground_indicies]
        
        #Store a mapping back to the original indicies for later reference
        self.pixel_to_original_map = foreground_indicies.tolist()
        
    def __convert_colour_space(self, rgb_array):
        """
        Convert RGB array to specified colour space
        
        Parameters
        ----------
        rgb_array : array
            RGB pixel values in range [0,255]

        Raises
        ------
        ValueError
            If colour_space is not supported.

        Returns
        -------
        converted_array : ndarray
            Pixel values converted to the target colour space:
            -RGB : [0, 255] range
            -LAB : L*a*b* values
            -HSV : H[0,360], S&V[0,100] 
            -XYZ : XYZ values
            -LUV : L*u*v* values

        """
        
        #Normalize array between [0,1]
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
        Convert color values from current color space back to RGB for display.
        
        Parameters
        ----------
        color_values : ndarray
            Values in the current color space
            
        Raises
        ------
            ValueError
                If colour_space is not supported.
            
        Returns
        -------
        rgb_values : ndarray
            RGB values in range [0,255]
            
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
        Randomly selects k unique centroids from the input pixel array
        
        Selects k random pixels as initial centroids, ensuring no duplicates.
        Uses random sampling without replacement to guarantee uniqueness.

        Returns
        -------
        picked_values : list
            List of k centroid color values selected from input_array
        picked_indices : list
            List of k indices corresponding to the selected centroids in input_array
            
        Notes
        -----
        Current implementation could be optimized using random.sample

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
            raise ValueError("Attempting to pick too many centroids for given image size")

        
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
        Assign each pixel to the nearest centroid based on Euclidean distance in 
        colour space.

        Parameters
        ----------
        centroids : array-like
            List or array of current centroid coordinates, shape (k, 3)
            
        Returns
        -------
        pixels : ndarray
            Array of pixel values, shape (num_pixels, 3)
        nearest : ndarray
            Array of centroid indices, shape (n_pixels,), where nearest[i] 
            is the index of the closest centroid for pixel i

        """

        #Convert to NumPy arrays
        pixels = np.asarray(self.input_array)
        centroids = np.asarray(centroids)
    
        # Compute distances using NumPy
        distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
    
        # Pick nearest centroid index for each pixel
        nearest = np.argmin(distances, axis=1)
    
        #Recreate original structure of assignments
        #assignments = [[pixel.tolist(), idx] for pixel, idx in zip(pixels, nearest)]
    
        return pixels, nearest
    
    def update_centroids(self, pixels, nearest, current_centroids):
        """
        Update centroid positions as the mean of assigned pixels.
        
        Computes new centroid positions by averaging all pixels assigned to each
        centroid. If a centroid has no assigned pixels, it retains its previous position.

        Parameters
        ----------
        pixels : numpy.ndarray
            Array of pixel values, shape (n_pixels, 3)
        nearest : numpy.ndarray
            Array of centroid assignments for each pixel, shape (n_pixels,)
        current_centroids : array-like
            Current centroid coordinates, shape (k, 3)

        Returns
        -------
        new_centroids : ndarray
            Updated centroid coordinates based on mean of assigned pixels.
            If a centroid has no assigned pixels, it retains its previous position.

        """
        
        k = len(current_centroids)
        new_centroids = np.zeros((k, 3))
        
        # Sum pixels per centroid
        np.add.at(new_centroids, nearest, pixels)
        
        # Count pixels per centroid
        counts = np.bincount(nearest, minlength=k)
        
        # Avoid divide-by-zero for empty clusters
        mask = counts > 0
        current_centroids = np.asarray(current_centroids)
        new_centroids[mask] /= counts[mask][:, None]  # broadcast (m,1) with (m,3)
        new_centroids[~mask] = current_centroids[~mask]
        
        return new_centroids
                
    def kmeans_loop(self, max_iterations = 10, tolerance=1e-3, display_interval = 0):
        """
        Perform K-means clustering algortihm on the input array.
        
        Iteratively assigns pixels to nearest centroids and updates centroid positions
        until they convergence or a max number of iterations is reached.
        Optionally displays clustering progress at specified intervals.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations to run.
            default = 10.
        tolerance : float, optional
            Threshold for minimum centroid movement to determine convergence.
            default = 1e-3.
        display_interval : int, optional
            Number of iterations between visual outputs of centriod assignments.
            If display_interval = 0, no images are shown.
            default = 0.

        Returns
        -------
        centroids : ndarray
            Final centroid positions after convergence (or max iterations) with shape (k, 3).
        pixels : numpy.ndarray
            Pixel values used in clustering, shape (n_pixels, 3).
        nearest : numpy.ndarray
            Final assignment of each pixel to centroid index, shape (n_pixels,).
        centroid_indices : list
            Initial indicies of centroids.
        """
        
        #Only show if messages enabled
        if self.messages:
            print(f"Starting k-means clustering in {self.colour_space} color space:")
        
        #Initialize centroids
        centroids, centroid_indices = self.__initalize_centroids()
        
        for i in range(max_iterations):
            #Assign each pixel to nearest centroid
            pixels, nearest = self.assign_to_centroids(centroids)
            
            #Compute new centroids
            new_centroids = self.update_centroids(pixels, nearest, centroids)
            
            #Display the clustering progress
            if display_interval and i % display_interval == 0:
                self.visualize_clustering(pixels, nearest, centroids)
            
            #Check if centroids have converged
            if np.allclose(new_centroids, centroids, atol=tolerance):
                if self.messages:
                    print(f"Convergence reach after {i} iterations")
                break
            
            centroids = new_centroids
        else:
            print(f"Reached maximum iterations ({max_iterations})")
        
        self.final_centroids = centroids
        self.final_assignments = (pixels, nearest)
        
        self.wcss = self.calculate_wcss()
        
        return centroids, pixels, nearest, centroid_indices
    
    def calculate_wcss(self):
        """
        Calculate Within-Cluster Sum of Squares (WCSS) for the clustering results.
        
        WCSS measures the sum of squared distances from each pixel to its assigned
        centroid across all clusters. Lower values indicate tighter clusters.
        
        Returns
        -------
        wcss : float or None
            Within-cluster sum of squares value. Returns None if no clustering
            results are available (final_centroids or final_assignments is None).
            
        Notes
        -----
        Requires that kmeans_loop() has been run first to evaluate final_centroids
        and final_assignments attributes.
        """
        
        if self.final_centroids is None or self.final_assignments is None:
            return None
    
        # Separate pixels and their assignments into NumPy arrays
        # This is the corrected section.
        pixels_np, final_assignments_np = self.final_assignments
        pixels_np = np.array(pixels_np)
        final_assignments_np = np.array(final_assignments_np)
        final_centroids_np = np.array(self.final_centroids)

        wcss = 0
        
        # Loop through each cluster
        for k in range(len(final_centroids_np)):
            # Get pixels assigned to this cluster using boolean indexing
            cluster_pixels = pixels_np[final_assignments_np == k]
            
            if len(cluster_pixels) > 0:
                # Get the centroid for this cluster
                centroid = final_centroids_np[k]
                diffs = cluster_pixels - centroid
                # Calculate the sum of squared distances for this cluster
                wcss += np.sum(np.einsum('ij,ij->i', diffs, diffs))
                
        return wcss
        
    def visualize_clustering(self, pixels, nearest, current_centroids):
        """
        Visualize current clustering state with side-by-side comparison.
        
        Displays the original image alongside the current clustering result,
        where each pixel is colored according to its assigned centroid's color.
        Used during kmeans_loop() to show clustering progress.

        Parameters
        ----------
        pixels : numpy.ndarray
            Current pixel values, shape (n_pixels, 3)
        nearest : numpy.ndarray  
            Current centroid assignments, shape (n_pixels,)
        current_centroids : numpy.ndarray
            Current centroid positions, shape (k, 3)

        Returns
        -------
        None
            Displays matplotlib figure with original and clustered images
            
        Notes
        -----
        Handles both background filtering and non-filtering modes. When 
        ignore_background=True, background pixels are colored black and only
        foreground pixels show cluster colors.
        """
        
        #Generate distinct colours for each centroid.
        #colours = self.colour_map(saturation, value)
        rgb_centroids = self.__convert_back_to_rgb(np.array(current_centroids))
        
        #Make a copy of image so original is not modified.
        #temp_image = self.image.copy().reshape(-1, 3)
        temp_image = np.zeros((len(self.original_input_array), 3), dtype=np.uint8)
        
        if self.ignore_background:
            temp_image[:] = [0,0,0]
            
            for i, centroid_idx in enumerate(nearest):
                original_pixel_idx = self.pixel_to_original_map[i]
                temp_image[original_pixel_idx] = rgb_centroids[centroid_idx]
        else:
            for i, centroid_idx in enumerate(nearest):
                temp_image[i] = rgb_centroids[centroid_idx]
           
        temp_image = temp_image.reshape(self.image.shape)
        
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
        Visualize final clustering results using custom gradient colors.
        
        Creates a visualization where each cluster is displayed using colors
        from the self.colours attribute instead of the centroid colors.
        Shows original image and custom-colored clustering side by side.

        Returns
        -------
        None
            Displays matplotlib figure comparing original and custom-colored results
            
        Notes
        -----
        Requires that kmeans_loop() has been run first. Uses self.colours for
        cluster visualization instead of centroid colors. Prints error message
        if no clustering results are available.
        """
        
        if self.final_centroids is None or self.final_assignments is None:
            print("No clustering results available. Run kmeans_loop() first.")
            return
        
        _, nearest = self.final_assignments
        
        # Create output image using gradient colors
        if self.ignore_background:

            temp_image = np.zeros((len(self.original_input_array), 3), dtype=np.uint8)
            temp_image[:] = [0,0,0]
        
            for i, centroid_idx in enumerate(nearest):
                original_pixel_idx = self.pixel_to_original_map[i]
                temp_image[original_pixel_idx] = self.colours[centroid_idx]
            
        else:
            
            temp_image = np.zeros((len(nearest), 3), dtype=np.uint8)
            
            # Assign each pixel the gradient color of its centroid
            for i, centroid_idx in enumerate(nearest):
                temp_image[i] = self.colours[centroid_idx]
                
        # Reshape back to original image dimensions
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
        Extract dominant colors and their proportions from clustering results.
        
        Returns the final centroid colors converted to RGB format along with
        the percentage of the image that each color represents.
        
        Returns
        -------
        colors_rgb : list or None
            List of dominant colors in RGB format as [R, G, B] values.
            Returns None if no clustering results available.
        percentages : list or None
            List of percentages (0-100) indicating what proportion of the image
            each corresponding color represents. Returns None if no clustering 
            results available.
            
        Notes
        -----
        Requires that kmeans_loop() has been run first. Colors are returned
        in RGB format regardless of the color space used for clustering.
        """
        
        if self.final_centroids is None or self.final_assignments is None:
            print("No clustering results available. Run kmeans_loop() first.")
            return None, None
        
        #Unpack final_assignments
        _, nearest = self.final_assignments 
        
        # Count pixels per cluster
        cluster_counts = np.bincount(nearest, minlength=self.k)
    
        # Convert centroids to RGB
        colors_rgb = self.__convert_back_to_rgb(np.array(self.final_centroids))
    
        # Calculate percentages
        total_pixels = len(nearest)
        percentages = (cluster_counts / total_pixels) * 100
    
        return colors_rgb.tolist(), percentages.tolist()
    
    def compare_color_spaces(self, color_spaces=["RGB", "LAB", "HSV", "XYZ", "LUV"]):
        """
        Compare K-means clustering results across different color spaces.
        
        Performs clustering using the same parameters but different color spaces
        and displays the results side by side for visual comparison.

        Parameters
        ----------
        color_spaces : list, optional
            List of color space strings to compare. Default is 
            ["RGB", "LAB", "HSV", "XYZ", "LUV"].

        Returns
        -------
        None
            Displays matplotlib figure with original image and clustering results
            for each specified color space
            
        Notes
        -----
        Creates temporary KMeans_Image instances for each color space comparison.
        Each clustering uses the same k value and runs for 10 iterations.
        Results show how different color spaces affect clustering outcomes.
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

    

if __name__ == "__main__":
    start = time.time()
    k=2
    image_rgb = Image_To_Array("display_images/checkerboard.jpg", "RGB")
    #image_rgb.show_image()
    colour_object = Colour_Map_Object()
    colours=colour_object.gradient_colour_map(k, "snw")
    kmeans_rgb = KMeans_Image(image_rgb, k, colours=colours, ignore_background=False, messages = True)
    kmeans_rgb.kmeans_loop(tolerance=1e-3,max_iterations=10, display_interval=1)
    #kmeans_rgb.visualize_with_gradient_colors()
    #print(kmeans_rgb.wcss)
    #end=time.time()
    #print(end-start)