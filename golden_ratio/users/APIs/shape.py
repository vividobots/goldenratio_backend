import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import ImageFile
from keras.preprocessing.image import load_img,img_to_array
ImageFile.LOAD_TRUNCATED_IMAGES=True



loaded_model = tf.keras.models.load_model("face_shape_model.h5")


classes=['Heart', 'Oblong', 'Oval', 'Round', 'Square']


def predict_shape_face(file):
   img=load_img(file,target_size=(200,200))
   img=img_to_array(img)
   img=img.reshape(1,200,200,3)
   img=img.astype('float32')
   img/=255
   return classes[np.argmax(loaded_model.predict(img),axis=1)[0]]
   




