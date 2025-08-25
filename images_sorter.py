from k_means_clustering import KMeans_Image
from image_to_array import Image_To_Array
from colorsys import rgb_to_hsv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

from datasets import load_dataset


class Images_Sorter():
    
    def __init__(self, folder_path, image_type):
        """
        Initialize Images_Sorter for analyzing and sorting images by color properties.
        
        Automatically reads all images of the specified type from the given folder
        and converts them to Image_To_Array objects for processing.
        
        Parameters
        ----------
        folder_path : str
            Path to the folder containing images to be sorted
        image_type : str
            File extension of images to read (e.g., 'png', 'jpg', 'jpeg')
            
        Attributes Set
        --------------
        folder_path : str
            Stored folder path
        image_type : str
            Stored image file extension
        training_set : list
            List of Image_To_Array objects loaded from the folder
        """
        
        self.folder_path = folder_path
        self.image_type = image_type
        
        self.training_set = self.read_images_from_folder(folder_path, image_type)
        
    def read_images_from_folder(self, folder_path, image_type="png"):
        """
        Read all images of specified type from a local folder.
        
        Uses glob to find all files matching the pattern *.{image_type} in the 
        specified folder and converts each to an Image_To_Array object in RGB format.
        
        Parameters
        ----------
        folder_path : str
            Path to folder containing images
        image_type : str, optional
            File extension to search for. Default is "png".
            
        Returns
        -------
        images : list
            List of Image_To_Array objects, one for each image found in the folder
            
        Notes
        -----
        All images are loaded in RGB color space. The method will include all files
        matching the extension pattern, so ensure the folder contains only valid images.
        """
        
        images = []
        data_set = glob.glob(os.path.join(folder_path, f"*.{image_type}"))
        for image_path in data_set:
            images.append(Image_To_Array(image_path, "RGB"))
        return images
    
    def sort_by_temperature(self):
        """
        Sort images into warm and cool categories based on dominant colors.
        
        Performs K-means clustering (k=3) on each image to extract dominant colors,
        then calculates a warmth score based on red/yellow vs blue content.
        Images with warmth scores above 50 are classified as warm.
        
        Returns
        -------
        warm_images : list
            List of Image_To_Array objects classified as having warm color tones
        cool_images : list
            List of Image_To_Array objects classified as having cool color tones
            
        Notes
        -----
        Warmth scoring formula: (R + G/2 - B) * (percentage/100) summed across
        all dominant colors. The threshold of 50 can be adjusted based on desired
        sensitivity. Higher scores indicate warmer colors (more red/yellow).
        """
        
        warm_images = []
        cool_images = []
        
        for image in self.training_set:
            kmeans = KMeans_Image(image, k=3)
            kmeans.kmeans_loop()
            colors, percentages = kmeans.get_dominant_colors()
            
            if colors:
                # Calculate weighted average temperature
                warmth_score = 0
                for color, percentage in zip(colors, percentages):
                    r, g, b = color
                    # Simple warmth metric: more red/yellow = warmer
                    color_warmth = (r + g/2 - b) * (percentage/100)
                    warmth_score += color_warmth
                
                if warmth_score > 50:  # Define your threshold
                    warm_images.append(image)
                else:
                    cool_images.append(image)
        
        return warm_images, cool_images  
      
    def display_sorted_images(self, warm_images, cool_images):
        """
        Display warm and cool images in separate grid visualizations.
        
        Creates two matplotlib figure grids showing warm and cool images separately.
        Each grid displays up to 5 columns with as many rows as needed. Images
        are titled with their category and index number.
        
        Parameters
        ----------
        warm_images : list
            List of Image_To_Array objects classified as warm
        cool_images : list
            List of Image_To_Array objects classified as cool
            
        Returns
        -------
        None
            Displays matplotlib figures for warm and cool image grids
            
        Notes
        -----
        Assumes Image_To_Array objects have an rgb_image attribute for display.
        Grid layout is optimized for readability with maximum 5 columns per row.
        Empty lists will be skipped (no figure displayed).
        """
        
        # Display warm images
        if warm_images:
            n_warm = len(warm_images)
            cols = min(5, n_warm)  # Max 5 columns
            rows = (n_warm + cols - 1) // cols  # Calculate rows needed
            
            plt.figure(figsize=(15, 3*rows))
            plt.suptitle(f'Warm Images ({n_warm} total)', fontsize=16)
            
            for i, image in enumerate(warm_images):
                plt.subplot(rows, cols, i+1)
                plt.imshow(image.rgb_image)  # Assuming your Image_To_Array has rgb_image attribute
                plt.title(f'Warm {i+1}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Display cool images  
        if cool_images:
            n_cool = len(cool_images)
            cols = min(5, n_cool)
            rows = (n_cool + cols - 1) // cols
            
            plt.figure(figsize=(15, 3*rows))
            plt.suptitle(f'Cool Images ({n_cool} total)', fontsize=16)
            
            for i, image in enumerate(cool_images):
                plt.subplot(rows, cols, i+1)
                plt.imshow(image.rgb_image)
                plt.title(f'Cool {i+1}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            
    def sort_by_spectrum_enhanced(self):
        """
        Sort images by color spectrum using HSV-based color classification.
        
        Performs advanced color analysis using K-means clustering (k=5) with background
        filtering, then classifies each image's most dominant saturated color into
        spectrum categories (red, orange, yellow, green, blue, purple, magenta).
        
        Returns
        -------
        sorted_images : list
            List of dictionaries sorted by hue value, each containing:
            - 'image': Image_To_Array object
            - 'hue': Hue value in degrees (0-360) used for sorting
            - 'color_name': Color category name (e.g., 'red', 'blue')
            - 'dominant_color': RGB values [R, G, B] of the dominant color
            - 'percentage': Percentage of image this color represents
            - 'hsv': HSV values [H, S, V] of the dominant color
            
        Notes
        -----
        Color classification ranges:
        - Red: 0-15° and 330-360°
        - Orange: 15-45°
        - Yellow: 45-75°
        - Green: 75-150°
        - Blue: 150-250°
        - Purple: 250-290°
        - Magenta: 290-330°
        - Grayscale: Low saturation (<0.15) or value (<0.15)
        
        Images are scored by (percentage × saturation) to prioritize vibrant,
        dominant colors over muted or minor color components.
        """
    
        def classify_color(hue, saturation, value):
            """Classify color into spectrum categories"""
            # Handle grayscale (low saturation)
            if saturation < 0.15 or value < 0.15:
                return "grayscale", 999  # Put at end
            
            # Color ranges in hue degrees
            color_ranges = [
                (0, 15, "red"),
                (15, 45, "orange"), 
                (45, 75, "yellow"),
                (75, 150, "green"),
                (150, 250, "blue"),
                (250, 290, "purple"),
                (290, 330, "magenta"),
                (330, 360, "red")  # Wrap around
            ]
            
            for start, end, color_name in color_ranges:
                if start <= hue < end:
                    return color_name, hue
            
            return "red", hue  # Default
        
        image_data = []
        
        for image in self.training_set:
            kmeans = KMeans_Image(image, k=5, ignore_background=True)
            kmeans.kmeans_loop()
            colors, percentages = kmeans.get_dominant_colors()
            
            if colors:
                # Find most dominant saturated color (ignore grays)
                best_color = None
                best_score = -1
                
                for color, percentage in zip(colors, percentages):
                    r, g, b = [c/255.0 for c in color]
                    hsv = rgb_to_hsv(r, g, b)
                    
                    # Score based on percentage and saturation
                    score = percentage * hsv[1]  # Prefer saturated colors
                    
                    if score > best_score:
                        best_score = score
                        best_color = (color, hsv, percentage)
                
                if best_color:
                    color, hsv, percentage = best_color
                    hue = hsv[0] * 360
                    color_name, sort_hue = classify_color(hue, hsv[1], hsv[2])
                    
                    image_data.append({
                        'image': image,
                        'hue': sort_hue,
                        'color_name': color_name,
                        'dominant_color': color,
                        'percentage': percentage,
                        'hsv': hsv
                    })
        
        # Sort by hue
        return sorted(image_data, key=lambda x: x['hue'])
        
    def display_spectrum_sorted(self, sorted_images):
        """
        Display images sorted by color spectrum with color progression visualization.
        
        Creates two visualizations: a grid of sorted images labeled with color names
        and hue degrees, and a color palette showing the dominant color progression.
        
        Parameters
        ----------
        sorted_images : list
            List of image dictionaries from sort_by_spectrum_enhanced(), sorted by hue
            
        Returns
        -------
        None
            Displays two matplotlib figures:
            1. Grid of images sorted by spectrum with color labels
            2. Linear color palette showing dominant color progression
            
        Notes
        -----
        Grid layout uses up to 10 columns for better spectrum visualization.
        Color palette is displayed as a single row showing the natural color
        progression from red through the spectrum. Images should be pre-sorted
        by the sort_by_spectrum_enhanced() method for proper color ordering.
        """
        
        n_images = len(sorted_images)
        cols = min(10, n_images)
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=(20, 2*rows))
        plt.suptitle('Images Sorted by Color Spectrum (Red→Orange→Yellow→Green→Blue→Purple)', fontsize=16)
        
        for i, img_data in enumerate(sorted_images):
            plt.subplot(rows, cols, i+1)
            plt.imshow(img_data['image'].rgb_image)
            
            # Color name and hue as title
            hue = img_data['hue']
            color_name = img_data.get('color_name', f'Hue {hue:.0f}°')
            plt.title(f'{color_name}\n{hue:.0f}°', fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Also show color palette
        colors = [img_data['dominant_color'] for img_data in sorted_images]
        palette = np.array(colors).reshape(1, -1, 3) / 255.0
        
        plt.figure(figsize=(20, 2))
        plt.imshow(palette)
        plt.title('Dominant Color Progression')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    folder_path = "training_images/pixel_art_images/"
    sort = Images_Sorter(folder_path, "png")
    b = sort.sort_by_temperature()
    sort.display_sorted_images(b[0], b[1])
    spectrum_sorted = sort.sort_by_spectrum_enhanced()
    sort.display_spectrum_sorted(spectrum_sorted)

