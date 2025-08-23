from k_means_clustering import KMeans_Image
from image_to_array import Image_To_Array
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class K_Value_Optimizer:
    
    def __init__(self, training_set):
        self.training_set = training_set
    
    def elbow_method(self):
        pass
    
    def curve_fit(self):
        pass
    
    def predict_k(self):
        pass


#Notes:
#pixel art and sprite art images have black backgrounds
#this massively effects k-means clustering, and so will likely need to ignore black pixels
#real world images take waaaaaaaaaaaay too long to deal with currently
#massive optimizations are needed, some options include taking 1 in every 4 pixels
#seems like cv2 can be used for resizing




x=[];y=[]

for i in np.arange(0,1000,0.1):
    y.append(1*np.log(1+i) + 1)
    x.append(i)
    
    
plt.plot(x,y)
plt.show()


