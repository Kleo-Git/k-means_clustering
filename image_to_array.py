import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from PIL import Image


class Image_To_Array:
    """
    A utility class that converts an image into pixel arrays in different colours space.
    
    The class lads an image from disk, converts it to the desired colour space, and
    provides both 3D image arrays (height x width x channels) and flattened 2D
    arrays (num_pixels x channels).
    """
    
    def __init__(self, filepath, colour_space = "RGB"):
        """
        Initialize the Image_To_Array object.

        Parameters
        ----------
        filepath : str
            Path to the image file to be loaded.
        colour_space : str, optional
            Colour space to convert to. 
            Options include:
                'RGB' - Red, Green, Blue (default)
                'LAB' - Lightness, A*, B* (perceptually uniform)
                'HSV' - Hue, Saturation, Value  
                'XYZ' - CIE XYZ colour space
                'LUV' - CIE LUV colour space
            
        Attributes
        ----------
        filepath : str
            Path of input image.
        colour_space : str
            Target colour space of the processed image.
        rgb_image : ndarray
            Original image stored as a NumPy array in RGB format, shape (H,W,3)
        image_shape : tuple
            Shape of the input image as (H,W,Channels)
        image : ndarray
            Image converted into selected colour space, shape (H,W,3)
        rgb_array : ndarray
            Flattened array of RGB pixel values.
        colour_array : ndarray
            Flattened array of pixel values in target colour space.
            
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
        
    #Private method to convert image as RGB array.
    def __load_rgb_image(self):
        """
        Convert the input image file into an RGB NumPy array.
        
        Uses OpenCV to read the image in BGR format and converts
        to RGB.

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
        Convert the loaded RGB image to the target colour space.
        
        Allows for the following colour space:
            - 'RGB' : Returns a copy of the original RGB image
            - 'LAB' : CIELAB (uses skimage, expects float in [0, 1])
            - 'HSV' : OpenCV HSV (H: 0-179, S/V: 0-255)
            - 'XYZ' : CIE 1931 XYZ (skimage, scaled by 100)
            - 'LUV' : CIE 1976 L*u*v* (skimage, expects float in [0, 1])
            
        Normalization : For skimage conversions (LAB, XYZ, LUV) the RGB
        image is first normalised to [0,1].

        Returns
        -------
        converted : ndarray
            Image converted to the specified colour space, with shape (H,W,3)
            dtype=float32 for most colour spaces.
            
        Raises
        ------
        ValueError
            If an unsupported colour space is specified.

        """
        if self.colour_space == 'RGB':
            return self.rgb_image.copy()
        
        #Normalize RGB to [0,1] for skimage conversions
        rgb_normalized = self.rgb_image.astype(np.float32) / 255.0
        
        #Check all options and convert to desired colour space, handled by skimage color
        #or cv2 for HSV colour space.
        if self.colour_space == 'LAB':
            converted = color.rgb2lab(rgb_normalized)
            
        elif self.colour_space == 'HSV':
            # OpenCV HSV ranges: H 0-179, S/V 0-255
            converted = cv.cvtColor(self.rgb_image, cv.COLOR_RGB2HSV).astype(np.float32)
            
        elif self.colour_space == 'XYZ':
            converted = color.rgb2xyz(rgb_normalized) * 100  # Scale for better clustering
            
        elif self.colour_space == 'LUV':
            converted = color.rgb2luv(rgb_normalized)
            
        else:
            raise ValueError(f"Unsupported colour space: {self.colour_space}")
        
        return converted
    
    #Flatten to combined array each pixel has 3 numbers that represent RGB
    def __flatten_image(self, img):
        """
        Flatten a 3D image array (H,W,Channels) into a 2D list of pixel values.
        
        Each pixel is represented as a list of three numbers corresponding 
        to its colour channels (e.g. [R,G,B] for RGB or [L,A,B] for LAB).
        
        Parameters
        ----------
        img : ndarray
            The image as a NumPy array of shape (height, width, 3).

        Returns
        -------
        img : list of list[int or float]
            Flattened list where each element is a 3-element list for a pixel.
            Total length of the list will be height*width.

        """
        #Use numpy to efficiently reshape and convert to list
        return img.reshape(-1, 3).tolist()
    
    def show_image(self):
        """
        Display the current image using MatPlotLib.
        
        Assumes the image is in a displayable format, uses 'self.image' for display.

        Returns
        -------
        AttributeError
            If the image data is not loaded.

        """
        try:
            import matplotlib.pyplot as plt # Import matplotlib locally for display
            plt.imshow(self.img_data)
            plt.axis('off')
            plt.show()
        except:
            print("No image data to display.")
        
    def show_colour_space_comparison(self):
        """
        Display the same image in multiple colour spaces for visual comparison.
        
        Displays the 5 supported colour spaces side by side:
        "RGB", "LAB", "HSV", "XYZ", "LUV"
        Converts images back to RGB for display purposes.
        
        Notes
        -----
        - HSV is converted using OpenCV
        - LAB, XYZ and LUV are converted back to RGB using skimage.
        
        
        Returns
        -------
        None.

        """
        #Supported colour spaces
        colour_spaces = ["RGB", "LAB", "HSV", "XYZ", "LUV"]
        #Plotting images in given size
        fig, axes = plt.subplots(1, len(colour_spaces), figsize=(20, 4))
        
        for i, space in enumerate(colour_spaces):
            #Create temporary image object with different colour space
            temp_img = Image_To_Array(self.filepath, space)
            
            #Handle display conversion
            if space == 'RGB':
                display_img = temp_img.image
            elif space == 'HSV':
                #Convert back to RGB for display
                display_img = cv.cvtColor(temp_img.image.astype(np.uint8), cv.COLOR_HSV2RGB)
            else:
                #For LAB, XYZ, LUV convert back to RGB for display
                if space == 'LAB':
                    display_img = color.lab2rgb(temp_img.image)
                elif space == 'XYZ':
                    display_img = color.xyz2rgb(temp_img.image / 100)
                elif space == 'LUV':
                    display_img = color.luv2rgb(temp_img.image)
                display_img = np.clip(display_img * 255, 0, 255).astype(np.uint8)
            
            #Handle image labelling
            axes[i].imshow(display_img)
            axes[i].set_title(f'{space}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_color_info(self):
        """
        Retrieve data about the current image and its colour space.
        
        Returns
        -------
        info : dict
            Dictionary containing:
                - "colour_space" : str, current colour space
                - "image_shape" : tuple, shape of the image (H,W,Channels)
                - "pixel_count" : int, total number of pixels
                - "ranges" : str, typical value ranges for each channel in the colour space
        """
        info = {
            'colour_space': self.colour_space,
            'image_shape': self.image_shape,
            'pixel_count': len(self.colour_array)
        }
        
        #Typical value ranges for given colour space and convention.
        if self.colour_space == 'RGB':
            info['ranges'] = 'R,G,B: 0-255'
        elif self.colour_space == 'LAB':
            info['ranges'] = 'L: 0-100, A,B: -127 to +127'
        elif self.colour_space == 'HSV':
            info['ranges'] = 'H: 0-179, S,V: 0-255 (OpenCV format)'
        elif self.colour_space == 'XYZ':
            info['ranges'] = 'X,Y,Z: 0-100+ (scaled)'
        elif self.colour_space == 'LUV':
            info['ranges'] = 'L: 0-100, U,V: varies'
            
        return info
    

if __name__ == "__main__":
    
    #File paths for test images
    image_path = "images/test_image.jpg"
    image_path2 = "images/test_image2.jpg"
    image_path3 = "images/test_image3.jpg"
    image_path4 = "images/lion_test.jpg"
    
    
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
    
    image_rgb = Image_To_Array(image_path4, "RGB")
    image_rgb.show_colour_space_comparison()
