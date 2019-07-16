from keras import backend as k 
from keras.models import *
from keras.optimizers import *
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, UpSampling2D, ZeroPadding2D

from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils import get_file
from keras.layers import Permute, Reshape, Activation

# from datasets import CONFIG
# from sklearn.utils.extmath import softmax
def softmax(x, restore_shape=True):
    """
    Softmax activation for a tensor x. No need to unroll the input first.

    :param x: x is a tensor with shape (None, channels, h, w)
    :param restore_shape: if False, output is returned unrolled (None, h * w, channels)
    :return: softmax activation of tensor x
    """
    _, c, h, w = x._keras_shape
    x = Permute(dims=(2, 3, 1))(x)
    x = Reshape(target_shape=(h * w, c))(x)

    x = Activation('softmax')(x)

    if restore_shape:
        x = Reshape(target_shape=(h, w, c))(x)
        x = Permute(dims=(3, 1, 2))(x)

    return x

#model
def get_dilation_model(input_shape, apply_softmax, input_tensor, classes):
    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor,shape=input_shape)
        else:
            model_in = input_tensor
    h = Conv2D(filters=64,kernel_size=(3, 3), activation="relu", name="conv1_1")(model_in)
    h = Conv2D(filters=64,kernel_size=(3, 3), activation="relu", name="conv1_2")(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Conv2D(filters=128,kernel_size=(3, 3), activation="relu", name="conv2_1")(h)
    h = Conv2D(filters=128,kernel_size=(3, 3), activation="relu", name="conv2_2")(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Conv2D(filters=256,kernel_size=(3, 3), activation="relu", name="conv3_1")(h)
    h = Conv2D(filters=256,kernel_size=(3, 3), activation="relu", name="conv3_2")(h)
    h = Conv2D(filters=256,kernel_size=(3, 3), activation="relu", name="conv3_3")(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Conv2D(filters=512,kernel_size=(3, 3), activation="relu", name="conv4_1")(h)
    h = Conv2D(filters=512,kernel_size=(3, 3), activation="relu", name="conv4_2")(h)
    h = Conv2D(filters=512,kernel_size=(3, 3), activation="relu", name="conv4_3")(h)
    h = Conv2D(filters=512,kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = Conv2D(filters=512,kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = Conv2D(filters=512,kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = Conv2D(filters=4096,kernel_size=(3, 3), dilation_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Conv2D(filters=4096,kernel_size=(1, 1), activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Conv2D(filters=classes,kernel_size=(1, 1), name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), dilation_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), dilation_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), dilation_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(32, 32))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), dilation_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)
    h = ZeroPadding2D(padding=(64, 64))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), dilation_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=classes,kernel_size=(3, 3), activation='relu', name='ctx_fc1')(h)
    h = Conv2D(filters=classes,kernel_size=(1, 1), name='ctx_final')(h)
    # the following two layers pretend to be a Deconvolution with grouping layer.
    # never managed to implement it in Keras
    # since it's just a gaussian upsampling trainable=False is recommended
    h = UpSampling2D(size=(8, 8))(h)
    logits = Conv2D(filters=classes,kernel_size=(16,16),padding='same',use_bias=False,trainable=False,name='ctx_upsample')(h)

    if apply_softmax:
        model_out = softmax(logits)
       
    else:
        model_out = logits
    # print(model_out)
    # print(logits)
    # print(model_out)
    model = Model(name="dilation_model", inputs=model_in, outputs=model_out)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
# input_shape = (512,512,1)
# model = get_dilation_model(input_shape,True,None,20)
# model.summary()