import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from keras.applications import VGG16
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.backend import clear_session
import tensorflow as tf

class CallVGG:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)
        print('load model done')
        self.graph = tf.get_default_graph()

    def doPredict(self):
        with self.graph.as_default():
            img_path = '/home/workstation/FACE_ID/FACE_IMAGE/ALL/Images/BACH/1.jpg'
            print(img_path)
            img = image.load_img(img_path, target_size=(224, 224))
            print(1)
            x = image.img_to_array(img)
            print(2)
            x = np.expand_dims(x, axis=0)
            print(3)
            x = preprocess_input(x)
            print(4)
            features = self.model.predict(x)
            print(5)
            print('predict done')
            print(features)
            print('clear session done')
            return features

cdef public object createVGG():
    print('create instance')
    return CallVGG()

cdef public object callPredict(object p):
    print('call predict')
    return p.doPredict()
