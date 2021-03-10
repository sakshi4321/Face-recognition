import os
import keras
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D
from keras_vggface import utils
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Lambda
from keras.preprocessing import image
from keras.engine.topology import Layer
import h5py

def cosine_model_vgg():
    
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))
    
    base_model = VGGFace(model='vgg16',include_top=False, input_shape=(224, 224, 3))


    for x in base_model.layers[:-3]:
        x.trainable = True

    #siamese network
    x1 = base_model(input_1)
    x2 = base_model(input_2)
    

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([x1, x2])
    
    x = Concatenate(axis=-1)([x5,x4, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc',auroc], optimizer=Adam(0.00001))

    model.summary()

    return model



def cosine_distance(vests):

    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)
    
def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    
def cos_dist_output_shape(shapes):

    shape1, shape2 = shapes
    return (shape1[0],1)
    
def main():
    
    model=cosine_model_vgg()
    #load images
    img1=image.load_img(sys.argv[1],target_size=(224,224,3))
    img2=image.load_img(sys.argv[2],target_size=(224,224,3))
    img1 = np.array(img1).astype(np.float)
    img1 = img1[np.newaxis,... ]
    img2 = np.array(img2).astype(np.float)
    img2 = img2[np.newaxis,... ]
    #preprocessing images
    xv1=x = utils.preprocess_input(img1, version=1)
    xv2 = utils.preprocess_input(img2, version=1)
    pred = model.predict([xv1,xv2]).ravel()
    print("Similarity between images:")
    print(pred)
if __name__ == "__main__":
    main()
  
