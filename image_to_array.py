import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import color


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
        self.colour_space = colour_space.upper()

        self.rgb_image = self.__load_rgb_image()
        self.image_shape = self.rgb_image.shape
        
        self.image = self.__convert_colour_space()
        
        self.rgb_array = self.__flatten_image(self.rgb_image)
        self.colour_array = self.__flatten_image(self.image)
        
    #Private method to convert image to array
    def __load_rgb_image(self):
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
    
    def __convert_colour_space(self):
        """
        

        Returns
        -------
        None.

        """
        if self.colour_space == 'RGB':
            return self.rgb_image.copy()
        
        rgb_normalized = self.rgb_image.astype(np.float32) / 255.0
        
        if self.colour_space == 'LAB':
            converted = color.rgb2lab(rgb_normalized)
            
        elif self.colour_space == 'HSV':
            # Use OpenCV for HSV (different from skimage HSV)
            converted = cv.cvtColor(self.rgb_image, cv.COLOR_RGB2HSV).astype(np.float32)
            
        elif self.colour_space == 'XYZ':
            converted = color.rgb2xyz(rgb_normalized) * 100  # Scale for better clustering
            
        elif self.colour_space == 'LUV':
            converted = color.rgb2luv(rgb_normalized)
            
        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")
        
        return converted
    
    #Flatten to combined array each pixel has 3 numbers that represent RGB
    def __flatten_image(self, img):
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
        #full_list = []
        #for i in range(self.image.shape[0]):
        #    for j in range(self.image.shape[1]):
        #        full_list.append(self.image[i][j].tolist())
        return img.reshape(-1, 3).tolist()
    
    def show_image(self):
        """
        Displays the image using matplotlib.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.image)
        plt.title(f'Image (displayed in RGB, processed in {self.color_space})')
        plt.axis('off')
        plt.show()
        
    def show_colour_space_comparison(self):
        """
        

        Returns
        -------
        None.

        """
        color_spaces = ["RGB", "LAB", "HSV", "XYZ", "LUV"]
        fig, axes = plt.subplots(1, len(color_spaces), figsize=(20, 4))
        
        for i, space in enumerate(color_spaces):
            # Create temporary image object with different color space
            temp_img = Image_To_Array(self.filepath, space)
            
            # For display, we need to handle different color spaces differently
            if space == 'RGB':
                display_img = temp_img.image
            elif space == 'HSV':
                # Convert back to RGB for display
                display_img = cv.cvtColor(temp_img.image.astype(np.uint8), cv.COLOR_HSV2RGB)
            else:
                # For LAB, XYZ, LUV - convert back to RGB for display
                if space == 'LAB':
                    display_img = color.lab2rgb(temp_img.image)
                elif space == 'XYZ':
                    display_img = color.xyz2rgb(temp_img.image / 100)
                elif space == 'LUV':
                    display_img = color.luv2rgb(temp_img.image)
                display_img = np.clip(display_img * 255, 0, 255).astype(np.uint8)
            
            axes[i].imshow(display_img)
            axes[i].set_title(f'{space}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_color_info(self):
        """
        Get information about the current color space.
        
        Returns
        -------
        dict
            Information about color space ranges and characteristics.
        """
        info = {
            'color_space': self.color_space,
            'image_shape': self.image_shape,
            'pixel_count': len(self.color_array)
        }
        
        if self.color_space == 'RGB':
            info['ranges'] = 'R,G,B: 0-255'
        elif self.color_space == 'LAB':
            info['ranges'] = 'L: 0-100, A,B: -127 to +127'
        elif self.color_space == 'HSV':
            info['ranges'] = 'H: 0-179, S,V: 0-255 (OpenCV format)'
        elif self.color_space == 'XYZ':
            info['ranges'] = 'X,Y,Z: 0-100+ (scaled)'
        elif self.color_space == 'LUV':
            info['ranges'] = 'L: 0-100, U,V: varies'
            
        return info
    

if __name__ == "__main__":
    
    #File paths for test images
    image_path = "images/test_image.jpg"
    image_path2 = "images/test_image2.jpg"
    image_path3 = "images/test_image3.jpg"
    
    
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
    
    image_rgb = Image_To_Array(image_path, "RGB")
