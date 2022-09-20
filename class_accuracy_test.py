import sys
import os
import numpy as np
from keras.models import load_model
from model_build import *

def Precision(y_true, y_pred):
    y_true = 1 - y_true
    y_pred = 1 - y_pred
    tp = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = np.sum(np.round(np.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + 0.000001)
    return precision

def Recall(y_true, y_pred):
    y_true = 1 - y_true
    y_pred = 1 - y_pred
    tp = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = np.sum(np.round(np.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + 0.000001)
    return recall

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + 0.000001))
    return f1

def class_test():
    np.random.seed(6)
    (x_train, y_train), (x_test, y_test) = load_data_2()
    (x_train, y_train), (x_test, y_test) = preprocess_2(x_train, y_train, x_test, y_test)
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]

    num_classes=4

    filepath = '/root/compare_lightweight/saved_models_SRnet_0.9759/model_044-0.97593.h5'
    #1
    modified_mobilenet = Modified_mobilenet()
    modified_mobilenet.load_weights(filepath)
    modified_mobilenet.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #2
    # mobilenet_small = Mobilenetv3_small()
    # mobilenet_small.load_weights(filepath)
    # mobilenet_small.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #3
    # mobilenet_large = Mobilenetv3_Large()
    # mobilenet_large.load_weights(filepath)
    # mobilenet_large.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #4
    # shufflenet = ShuffleNet(include_top=True,input_shape=(224,224,3), groups=3, pooling='avg', classes=config.class_num)
    # shufflenet.load_weights(filepath)
    # shufflenet.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #5
    # squeezenet = SqueezeNet(include_top=True, weights=None, input_shape=(224, 224, 3), pooling='avg',
    #                         classes=config.class_num)
    # squeezenet.load_weights(filepath)
    # squeezenet.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #6
    # mobilenetv1 = Mobilenetv1()
    # mobilenetv1.load_weights(filepath)
    # mobilenetv1.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #7
    # alexnet = Alexnet_dif()
    # alexnet.load_weights(filepath)
    # alexnet.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #8
    # vgg16 = Vgg16()
    # vgg16.load_weights(filepath)
    # vgg16.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # print("model load finish")
    #9
    # resnet50=Resnet50()
    # resnet50.load_weights(filepath)
    # resnet50.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #10
    # vgg19 = Vgg19()
    # vgg19.load_weights(filepath)
    # vgg19.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("model load finish")
    test_x = []
    test_y = []
    heavy_silt_x = x_test[0:458, :, :, :]
    test_x.append(heavy_silt_x)
    heavy_silt_y = y_test[0:458, :]
    test_y.append(heavy_silt_y)
    no_silt_x = x_test[458:932, :, :, :]
    test_x.append(no_silt_x)
    no_silt_y = y_test[458:932, :]
    test_y.append(no_silt_y)
    no_targets_x = x_test[932:1375, :, :, :]
    test_x.append(no_targets_x)
    no_targets_y = y_test[932:1375, :]
    test_y.append(no_targets_y)
    normal_silt_x = x_test[1375:1828, :, :, :]
    test_x.append(normal_silt_x)
    normal_silt_y = y_test[1375:1828, :]
    test_y.append(normal_silt_y)

    filename = 'class_result.txt'
    out_file = open(filename, 'a')
    name=['alexnet','resnet50','vgg16','vgg19']

    for j in range(4):
        scores = modified_mobilenet.evaluate(test_x[j],test_y[j],verbose=0)
        print(scores[1])
        out_file.write("--------------------------------------------\n")
        out_file.write(name[3]+'_heavy-ns-nt-normal accuracy:%0.6f\n' % (scores[1]))
    for i in range(4):
        y_pred=modified_mobilenet.predict(test_x[i],verbose=0)
        precision=Precision(test_y[i],y_pred)
        recall=Recall(test_y[i],y_pred)
        f1=F1(test_y[i],y_pred)
        out_file.write("--------------------------------------------\n")
        out_file.write(name[3] + 'precision:%0.4f\n' % precision)
        out_file.write(name[3] + 'recall:%0.4f\n' % recall)
        out_file.write(name[3] + 'F1 score:%0.4f\n' % f1)

    # y_pred = mobilenetv1.predict(x_test, verbose=0)
    # precision = Precision(y_test, y_pred)
    # recall = Recall(y_test, y_pred)
    # f1 = F1(y_test, y_pred)
    # print('precision:%0.4f\n' % precision, 'recall:%0.4f\n' % recall, 'F1 score:%0.4f\n' % f1)
    # out_file.write(name[5]+'precision:%0.4f\n' % precision)
    # out_file.write(name[5]+'small recall:%0.4f\n' % recall)
    # out_file.write(name[5]+'small F1 score:%0.4f\n' % f1)
    out_file.close()

if __name__ =='__main__':
    class_test()