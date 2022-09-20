import sys
import os
import numpy as np
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D,Dropout,Dense
from keras.models import Model
from lightmodel_build import *
from config import DefaultConfig

# from squeezenet_build import *
from adabound import AdaBound

def modified_mobilenet_train():
    np.random.seed(6)
    config=DefaultConfig

    (x_train, y_train), (x_test, y_test) = load_data_2()
    (x_train, y_train), (x_test, y_test) = preprocess_2(x_train, y_train, x_test, y_test, smooth_l=True)
    modified_mobilenet=Modified_mobilenet()
    # modified_mobilenet.summary()

    print("Modified loading finished")
    opt=keras.optimizers.Adam(lr=0.001)
    # opt1 = AdaBound(lr=1e-03,
    #                final_lr=0.1,
    #                gamma=1e-03,
    #                weight_decay=0.,
    #                amsbound=False)
    #opt2 = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
    modified_mobilenet.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model_modified_mobilenet,history=train(x_train, y_train, x_test, y_test,
                                    modified_mobilenet, config.batch_size,config.epochs,data_augmentation=True)
    print('modified_mobilenet training finished')
    scores=model_modified_mobilenet.evaluate(x_test, y_test, verbose=1)
    filename='result.txt'
    out_file= open(filename,'a')
    out_file.write("--------------------------------------------\n")
    print('modified_mobilenet test accuracy: ', scores[1])
    out_file.write('modified_mobilenet test accuracy: %0.6f\n' % (scores[1]))
    out_file.close()



if __name__ =='__main__':
    modified_mobilenet_train()
