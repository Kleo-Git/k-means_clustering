import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Image_To_Array:
    """
    A utility class that converts an image into an array of RGB values.
    
    The image is read from a file and stored both as a 3D NumPy array
    (height x width x 3) and as a flattened list (height*width, 3), of RGB values for each
    pixel, ordered from top-left to bottom-right pixel.
    """
    
    def __init__(self, filepath, colour_space = "RGB"):
        """
        Initialize the Image_To_Array object.

        Parameters
        ----------
        filepath : str
            Path to the image file to be loaded.
        color_space : str
            Color space to convert to. 
            Options include:
                'RGB' - Red, Green, Blue (default)
                'LAB' - Lightness, A*, B* (perceptually uniform)
                'HSV' - Hue, Saturation, Value  
                'XYZ' - CIE XYZ color space
                'LUV' - CIE LUV color space
            
        Attributes
        ----------
        filepath : str
            Path of input image.
        image : ndarray
            The image stored as a NumPy array in RGB format.
        rgb_array : list
            Flattened list of RGB values for each pixel in row-major order.
        image_shape : tuple
            Shape of the image as (height, width, channels).
            
        Returns
        -------
        None.

        """
        self.filepath = filepath
        self.image = self.__image_to_array()
        self.rgb_array = self.__flatten()
        self.image_shape = self.image.shape
        
    #Private method to convert image to array
    def __image_to_array(self):
        """
        Convert the input image file into an RGB NumPy array.
        
        Uses OpenCV to read the image in BGR format and converts
        to RGB

        Returns
        -------
        img : ndarray
            The image as a NumPy array of shape (height, width, 3).

        """
        img = cv.imread(self.filepath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
    
    #Flatten to combined array each pixel has 3 numbers that represent RGB
    def __flatten(self):
        """
        Flatten the image into a list of RGB values.
        
        Iterates row by row through the image and extracts
        the RGB values for each pixel.

        Returns
        -------
        full_list : list of list[int]
            The list where each element is [R,G,B] for a given pixel.
            The shape of the list will be (height*width, 3).

        """
        full_list = []
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                full_list.append(self.image[i][j].tolist())
        return full_list
    
    def show_image(self):
        """
        Displays the image using matplotlib.
        """
        plt.imshow(self.image)
        plt.show()
    
    

if __name__ == "__main__":
    
    #File paths for test images
    image_path = "test_image.jpg"
    image_path2 = "test_image2.jpg"
    image_path3 = "test_image3.jpg"
    
    
    #Reading image paths, and finding pixel values
    img = cv.imread(image_path)
    
    img2 = cv.imread(image_path2)
    
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    
    img3= cv.imread(image_path3)
    
    img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
    
    #Manually recolour for testing
    img3[0][0] = (0,255,0)
    img3[0][1] = (0,0,255)
    img3[1][0] = (255,0,255)
    img3[1][1] = (255,255,0)
    
    #Display image 3
    plt.imshow(img3)
    plt.show()
    
    #Create image to array object
    a = Image_To_Array(image_path3)
    
    #Print properties of object and show image
    print(a.rgb_array)
    print(a.image_shape)
    a.show_image()