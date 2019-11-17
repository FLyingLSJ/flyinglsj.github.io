from keras.applications import inception_v3
from keras import backend as K

K.set_learning_phase(0)

model = inception_v3.InceptionV3(weights="imagenet",
                                 include_top=False
                                 )
model.summary()
