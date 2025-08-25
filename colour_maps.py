import numpy as np
import matplotlib.pyplot as plt
import colorsys

class Colour_Map_Object():
    """
    A class for generating smooth colour gradients, and allowing discrete colours
    to be chosen from these gradients, or a HSV colour set.
    
    Provides a dictionary of common named colours (RGB) and a set of default
    gradient mappings that can be used for visualization.
    """
    
    #Common named colours stored in RGB format
    main_colours = {
        "black"   : [0,0,0],
        "white"   : [255,255,255],
        "red"     : [255,0,0],
        "lime"    : [0,255,0],
        "blue"    : [0,0,255],
        "yellow"  : [255,255,0],
        "cyan"    : [0,255,255],
        "magenta" : [255,0,255],
        "pink"    : [255,192,203],
        "silver"  : [192,192,192],
        "grey"    : [128,128,128],
        "maroon"  : [128,0,0],
        "olive"   : [128,128,0],
        "green"   : [0,128,0],
        "purple"  : [128,0,128],
        "teal"    : [0,128,128],
        "navy"    : [0,0,128],
        "orange"  : [255,165,0]
        } 
    
    default_gradients = {
        "jkr"      : [main_colours["black"], main_colours["white"]],
        "mjz"      : [main_colours["black"], main_colours["yellow"], main_colours["red"]],
        "exo"      : [main_colours["magenta"], main_colours["lime"], main_colours["cyan"], main_colours["red"]],
        "snw"      : [main_colours["navy"], main_colours["teal"], main_colours["cyan"], main_colours["white"]],
        "torq"     : [main_colours["black"], main_colours["white"], main_colours["yellow"]],
        "kia"      : [main_colours["teal"], main_colours["white"], main_colours["pink"]],
        "rgb"      : [main_colours["red"], main_colours["lime"], main_colours["blue"]],
        "fire"     : [main_colours["red"], main_colours["yellow"]],
        "sunlight" : [main_colours["orange"], main_colours["yellow"], main_colours["white"]],
        "ocean"    : [main_colours["navy"], main_colours["blue"], main_colours["cyan"]],
        "void"     : [main_colours["black"], main_colours["purple"], main_colours["black"]],
        "deep_sea" : [main_colours["black"], main_colours["navy"], main_colours["black"]]
        } 
    
    def __init__(self):
        """
        Initialize a Colour_Map_Object instance.
        
        Attributes
        ----------
        main_colours : dict[str, list[int]]
            Instance-specific copy of predefined RGB colour dictionary.
            
        Notes
        -----
        Copies the class-level "main_colours" dictionary to an instance
        level attribute. Allow modifying without altering global class
        definition.

        Returns
        -------
        None.

        """
        #Create a copy of main colours dictionary
        self.main_colours = self.__class__.main_colours.copy()
        
    def colour_gradient(self, selected_colours, detail=500):
        """
        Generate a smooth gradient between a set of colours.

        Parameters
        ----------
        selected_colours : list of list[int] or str
            -If a list : Each element should be an RGB triplet in
            the range [0,255]. Atleast two colours are required.
            -If a string : Treated as a key into "self.default_gradients" for 
            preset gradients.
        detail : int, optional
            The total number of interpolation steps across the gradient.
            Higher values will give a smoother gradient.
            default = 500

        Raises
        ------
        ValueError
            If a string is entered and not found in "self.default_gradients".

        Returns
        -------
        gradient : ndarray of shape(1, width, 3)
            A NumPy array representing the gradient in normalized RGB values
            [0,1]. Can be directly visualized with "plot_gradient".
            
        Notes
        -----
        If same colour is used for both entries, it will generate a solid block
        of that colour.
        Gradient is divded evenly across the chosen colours.

        """
        
        #If input is a string check if input matches predefined gradients
        if isinstance(selected_colours, str):
            if selected_colours not in self.default_gradients:
                raise ValueError(f"Unknown gradient preset '{selected_colours}'. "
                                 f"Available presets: {list(self.default_gradients.keys())}")
                
            #Select colours found in default gradients
            selected_colours = self.default_gradients[selected_colours]
            
            
        #Find number of colours
        num_colours = len(selected_colours)
        
        #Must have atleast 2 colours to generate a gradient
        if num_colours <= 1:
            raise ValueError(f"Need atleast 2 colours to make a gradient! Recieved {num_colours}.")
        
        #Divide total detail evenly across number of colours
        detail = detail // num_colours
        
        #Create empty list to store gradient values
        gradient=[]
        
        #Interpolate lineraly between each pair of consecutive colours
        for i in range(num_colours-1):
            gradient.append(np.linspace(selected_colours[i], selected_colours[i+1], detail)/255)
        
        #Reshape into (1,width,3) for compatability with imshow.
        gradient = np.reshape(gradient, [1,detail*(num_colours-1),3])     
        
        return gradient
    
    def gradient_colour_map(self, num_colours, selected_colours, detail=500):
        """
        Generate a discrete set of colours sampled from a smooth gradient.

        Parameters
        ----------
        num_colours : int
            The number of colours to sample from the gradient.
        selected_colours : list of list[int] or str
            - If list : Each element must be an RGB triplet in [0,255]
            - If string : Treated as key in "self.default_gradients"
              to use a predefined gradient.
        detail : int, optional
            Resolution of underlying gradient from which colours are sampled.
            Higher values produce smoother transitions.
            default = 500.

        Returns
        -------
        colours : list of list[float]
            A list of RGB colours (each as a list of floats in [0,255])
            sampled evenly across the gradient.

        """
        
        #Intialize a gradient based on selected colours
        col_gradient = self.colour_gradient(selected_colours, detail)
        
        colours=[]
        
        grad_len = col_gradient.shape[1]
        
        #Select evenly spaced indicies along the gradient
        colour_idxs = np.linspace(0, grad_len-1, num=num_colours, dtype=int)
        
        #Convert normalized [0,1] RGB values back to [0,255]
        for idx in colour_idxs:
            colours.append((255*col_gradient[0, idx]).tolist())
        
        return colours
        
    def hsv_colour_map(self, num_colours, saturation = 0.8, value = 0.9, hue_offset=0.0):
        """
        Generates a list of distinct RGB colours.
        
        Colours are created by evenly spacing hues around the HSV colour wheel,
        then converting them to RGB with specified saturation and value.
        

        Parameters
        ----------
        num_colours : int
            The number of distinct colours to generate.
        saturation : float, optional
            Saturation of the HSV colour (0 to 1). Higher values produce more vivid colours.
            default = 0.8
        value : float, optional
            Value/brightness of the HSV colour (0 to 1). Higher values produce brighter colours.
            default = 0.9
        hue_offset : float, optional
            Initial offset for the hue, allowing different sets of colours
            to be generated.
            default = 0.0

        Returns
        -------
        colours_corrected : list of list[int]
            List of RGB colours, each as [R,G,B] with integer values in [0,255].

        """
    
        colours=[]
        
        #Generates HSV colours evenly spaced by hue.
        for i in range(num_colours):
            hue = ((i / num_colours) + hue_offset) % 1.0
            colours.append(list(colorsys.hsv_to_rgb(hue, saturation, value)))
    
        colours_corrected = []
        
        #Convert to 0-255 RGB integers
        for i in range(len(colours)):
            colours_corrected.append([int(j * 255) for j in colours[i]])
            
        return colours_corrected
    
    def plot_gradient(self, gradient):
        """
        Display a colour gradient as an image.
        
        This function visualizes the given gradient array (generated from
        "colour_gradient"), using matplotlib's "imshow", which renders the
        data as a colour image.
        
        Parameters
        ----------
        gradient : array of shape (1, width, 3)
            A 3D array representing the gradient values to be displayed.
        """
        plt.imshow(gradient, aspect='auto', interpolation='nearest')
        plt.show()
    
    
if __name__ == "__main__":
    
    gradient = Colour_Map_Object()
    
    g = gradient.colour_gradient([gradient.main_colours["red"], gradient.main_colours["blue"]])
    
    #g = gradient.colour_gradient("torq")
    
    gradient.plot_gradient(g)
    
        
        
        
    
    
    
    
    
    
    
    