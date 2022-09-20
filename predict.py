import sys
import os
import numpy as np
from keras.models import load_model
from lightmodel_build import *

filepath = '/root/compare_lightweight/saved_models_SRnet_0.97593/model_044-0.9759.h5'
modified_mobilenet = Modified_mobilenet()
modified_mobilenet.load_weights(filepath)
modified_mobilenet.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

x_predict=[]
row,col=224,224
dir='root//classification_vvvip//predict_pics//heavy_silt//'
for i in os.listdir(dir):
    print(i)
    imgpath=os.path.join(dir,i)
    img = Image.open(imgpath)
    x = img_to_array(img)
    x_predict.append(x)

total_input = len(x_predict)
x_predict = np.array(x_predict)
x_predict = x_predict.reshape(total_input, row, col, 3)
x_predict = x_predict.astype('float32')
x_predict /= 255

y_pred = modified_mobilenet.predict(x_predict, verbose=0)
print(y_pred)
name=['heavy','no_silt','no_targers','normal']
filename = 'predict_result.txt'
out_file = open(filename, 'a')
out_file.write(name[3]+'predict:{}\r'.format(y_pred))
out_file.close()