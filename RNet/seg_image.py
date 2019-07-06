import keras
from keras.layers import Activation,Dense,Conv2D,MaxPooling2D,Conv2DTranspose,BatchNormalization,activations,Concatenate,ELU
from keras.models import Model,Input
import cv2
import numpy as np
def residual_network(shape):
    # Y last layer output M Channel, NxN Kernel Size, S Stride
    def residual_units(y,M,N,S,threeLayer,pool):
        shortcut = y
        if pool:
        	shortcut = MaxPooling2D(pool_size=(2, 2),strides=[2,2],padding='same')(shortcut)
        y = Conv2D(filters=M[0],kernel_size=(N[0],N[0]),strides=(S[0],S[0]),padding="same")(y)
        y = BatchNormalization()(y)
        #y = activations.elu(y,alpha=1.0)
        y = ELU()(y)
        y = Conv2D(filters=M[1],kernel_size=(N[1],N[1]),strides=(S[1],S[1]),padding="same")(y)
        y = BatchNormalization()(y)
        #y = activations.elu(y,alpha=1.0)
        y = ELU()(y)
        if threeLayer:
            y = Conv2D(filters=M[2],kernel_size=(N[2],N[2]),strides=(S[2],S[2]),padding="same")(y)
            y = BatchNormalization()(y)
            #y = activations.elu(y,alpha=1.0)
            y = ELU()(y)
        y = Concatenate()([shortcut,y])
        return y
    image = Input(shape)
    x = Conv2D(filters=64,kernel_size=[3,3],strides=[1,1],padding="same")(image)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same')(x)
    # Block 1
    shortcut1 = x
    shortcut1 = Conv2DTranspose(64,(4,4),strides=(2,2),padding='same',input_shape=(64,256,256))(shortcut1)
    x = residual_units(x,[128,128],[3,3],[2,1],False,True)
    x = residual_units(x,[128,128],[3,3],[1,1],False,False)
    x = residual_units(x,[128,128],[3,3],[1,1],False,False)
    # Block 2
    shortcut2 = x
    shortcut2 = Conv2DTranspose(64,(8,8),strides=(4,4),padding='same',input_shape=(128,128,128))(shortcut2)
    x = residual_units(x,[256,256],[3,3],[2,1],False,True)
    x = residual_units(x,[256,256],[3,3],[1,1],False,False)
    x = residual_units(x,[256,256],[3,3],[1,1],False,False)
    # Block 3
    x = residual_units(x,[512,512],[3,3],[1,1],False,False)
    x = residual_units(x,[512,512],[3,3],[1,1],False,False)
    x = residual_units(x,[512,512],[3,3],[1,1],False,False)
    x = residual_units(x,[512,512],[3,3],[1,1],False,False)
    x = residual_units(x,[512,512],[3,3],[1,1],False,False)
    x = residual_units(x,[512,512],[3,3],[1,1],False,False)
    # Block 4
    x = residual_units(x,[512,1024],[3,3],[1,1],False,False)
    x = residual_units(x,[512,1024],[3,3],[1,1],False,False)
    x = residual_units(x,[512,1024],[3,3],[1,1],False,False)
    # Block 5
    x = residual_units(x,[512,1024,2048],[1,3,1],[1,1,1],True,False)
    # Block 6
    x = residual_units(x,[1024,2048,4096],[1,3,1],[1,1,1],True,False)
    #Deconv
    x = Conv2DTranspose(64,(16,16),strides=(8,8),padding='same',input_shape=(4096,64,64))(x)
    #Fusion
    x = keras.layers.add([x,shortcut1,shortcut2])
    #Conv
    x = Conv2D(filters=32,kernel_size=[3,3],strides=[1,1],padding="same")(x)
    #Conv
    x = Conv2D(filters=16,kernel_size=[3,3],strides=[1,1],padding="same")(x)
    #Softmax
    x = Activation('softmax')(x)
    model = Model(inputs=image,outputs=x)
    return model


input_shape=(512,512,1)

model = residual_network(input_shape)
model.summary()