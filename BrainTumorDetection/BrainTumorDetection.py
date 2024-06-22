import cv2
from keras.models import load_model
from PIL import Image
import numpy as np


class BrainTumorDetection:
    def __init__(self, size):
        self.modelBin = load_model("BrainTumorDetection/BrainTumor10.h5")
        self.modelCat = load_model("BrainTumorDetection/BrainTumorCategorical10.h5")
        self.size = size 
    
    def PrepareImage(self, image):
        image = cv2.imread(image)
        image = Image.fromarray(image)
        image = image.resize((self.size, self.size))
        image = np.array(image)
        
        input = np.expand_dims(image, axis=0)
        result = self.modelBin.predict_step(input)
        result = result.numpy()
        print(result[0, 0])

test = BrainTumorDetection(size = 64)
test.PrepareImage("C:\\Users\\mabuza\\Desktop\\Machine-Learning\\BrainTumorDetection\\pred\\pred2.jpg") # There is tumor = 1
test.PrepareImage("C:\\Users\\mabuza\\Desktop\\Machine-Learning\\BrainTumorDetection\\pred\\pred0.jpg") # No tumor = 0