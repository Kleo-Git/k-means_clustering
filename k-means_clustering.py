from image_to_array import Image_To_ArrayRGB
import numpy as np
import random as rd
import cv2
import matplotlib.pyplot as plt


def initalize_centroids(input_array, k):
    
    #Total amount of pixels will equal length of array, given current setup
    num_pixels = len(input_array)
    
    picked_values = []
    
    #If k is larger then number of pixels would cause an infinite loop
    if k > num_pixels:
        print("Attempting to pick too many centroids for given image size")
        return
    
    #Should be much more efficient to use this.
    #random.sample(range(num_pixels), k)
    
    counter = 0
    while counter < k:
        pixel = rd.randint(0,num_pixels-1)
        #Prevent allowing for repeated pixels for a given centroid
        if pixel not in picked_values:
            picked_values.append(pixel)
            counter+=1
            
    return picked_values

def calculate_distance(position1, position2):
    #Calculates the squared sum of the differences in x,y,z positions.
    squared_sum = sum((a-b) ** 2 for a, b in zip(position1,position2))
    
    #Square root the squared sum to get total distance using pythagoras' theorem.
    distance = np.sqrt(squared_sum)
    
    #Return distance
    return distance
        
def k_means(input_array, k):
    
    centroids = initalize_centroids(input_array, k)

    print("a", centroids)
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    
    new_list = []
    
    for i in range(len(input_array)):
        shortest_distance = float("inf")
        associated_centroid = 0
        
        for centroid in centroids:
            
            current_distance = calculate_distance(input_array[i], input_array[centroid])
            
            if current_distance < shortest_distance:
                shortest_distance = current_distance
                associated_centroid = centroid
            
        new_list.append([input_array[i], associated_centroid])
        
    img = cv2.imread("images/test_image4.jpg")
    for centroid in centroids:
        x = (centroid-1) // 4
        #print(x)
        y = (centroid-1) % 4
        img[x][y] = (0,0,0)
    plt.imshow(img)
    plt.show()
   
    
    return new_list

def colour_centroids(image):
    
    pass

a = Image_To_ArrayRGB("images/test_image4.jpg")
#a.show_image()
#print(len(a.rgb_array))

print(a.image)

print("AUWHDUAWHDUAWH")

img = cv2.imread("images/test_image4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img)
print(img.shape)


plt.show()

b = k_means(a.rgb_array, 3)

#print(b)

#Use hue transformations to dynamically come up with colour maps based on k
#Random pseudocode
#for cluster i in range(k):
#hue = i / k
#saturation = 0.8
#value = 0.9
#convert HSV to RGB






