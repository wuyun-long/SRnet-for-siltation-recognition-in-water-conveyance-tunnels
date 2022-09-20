import os
from PIL import Image
from keras.preprocessing.image import  img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
import keras
from config import DefaultConfig
from cosine_annealing import CosineAnnealingScheduler
from keras.callbacks import ReduceLROnPlateau
import numpy as np
from mobilenetv3 import *
from keras.datasets import cifar10
from mixup_generator import MixupGenerator
import time
config=DefaultConfig


def smooth_labels(labels, factor=0.2):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])

    # returned the smoothed labels
    return labels

def load_data_3():
    x_train=[]
    x_valid=[]
    x_test=[]
    img_label=0
    y_train=[]
    y_valid=[]
    y_test=[]
    row,col=224,224
    DATA_DIR='/root/silt_data/'
    for i in os.listdir(DATA_DIR):
        path=DATA_DIR+i
        for j in os.listdir(path):
            imgpath = path+'/'+j
            img = Image.open(imgpath)
            # resize_img = img.resize((row, col))
            x = img_to_array(img)
            rua=np.random.randint(100)
            if rua<20:
                x_valid.append(x)
                y_valid.append(img_label)
            elif rua<40:
                x_test.append(x)
                y_test.append(img_label)
            else:
                x_train.append(x)
                y_train.append(img_label)
        img_label+=1
    total_input = len(x_train)
    x_train = np.array(x_train)
    x_train = x_train.reshape(total_input, row, col, 3)
    y_train = np.array(y_train)
    total_input_valid = len(x_valid)
    x_valid = np.array(x_valid)
    x_valid = x_valid.reshape(total_input_valid,row,col,3)
    y_valid = np.array(y_valid)
    total_input_test = len(x_test)
    x_test = np.array(x_test)
    x_test = x_test.reshape(total_input_test, row, col, 3)
    y_test = np.array(y_test)

    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'valid samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_valid,y_valid), (x_test, y_test)

def load_data_2():
    x_train=[]
    x_test=[]
    img_label=0
    y_train=[]
    y_test=[]
    row,col=224,224
    DATA_DIR='/root/silt_data/'
    for i in os.listdir(DATA_DIR):
        path=DATA_DIR+i
        for j in os.listdir(path):
            imgpath = path+'/'+j
            img = Image.open(imgpath)
            # resize_img = img.resize((row, col))
            x = img_to_array(img)
            rua=np.random.randint(100)
            if rua<30:
                x_test.append(x)
                y_test.append(img_label)
            else:
                x_train.append(x)
                y_train.append(img_label)
        img_label+=1
    total_input = len(x_train)
    x_train = np.array(x_train)
    x_train = x_train.reshape(total_input, row, col, 3)
    y_train = np.array(y_train)
    total_input_test = len(x_test)
    x_test = np.array(x_test)
    x_test = x_test.reshape(total_input_test, row, col, 3)
    y_test = np.array(y_test)

    print(x_train.shape[0], 'train samples')

    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def preprocess_3(x_train, y_train,x_valid,y_valid, x_test, y_test, smooth_l=False ):
    num_classes = 4
    # Convert class vectors to binary class matrices
    if smooth_l:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_train = smooth_labels(y_train,0.2)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
        y_valid = smooth_labels(y_valid,0.2)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_test = smooth_labels(y_test,0.2)
    else:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    # preprocess data
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_valid /= 225
    x_test /= 255
    # if substract_pixel_mean:
    #     x_train_mean = np.mean(x_train, axis=0)
    #     x_train -= x_train_mean
    #     x_test -= x_train_mean
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def preprocess_2(x_train, y_train, x_test, y_test, smooth_l=False ):
    num_classes = 4
    # Convert class vectors to binary class matrices
    if smooth_l:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_train = smooth_labels(y_train,0.2)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_test = smooth_labels(y_test,0.2)
    else:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    # preprocess data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # if substract_pixel_mean:
    #     x_train_mean = np.mean(x_train, axis=0)
    #     x_train -= x_train_mean
    #     x_test -= x_train_mean
    return (x_train, y_train), (x_test, y_test)

def preprocess_cifar10(x_train, y_train, x_test, y_test, substract_pixel_mean=False):
    num_classes = 10
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # preprocess data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    if substract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    return (x_train, y_train), (x_test, y_test)

def train(x_train, y_train, x_test, y_test, model, batch_size, epochs, data_augmentation):
    if not data_augmentation:
        print('Not using data augmentation.')
        filepath = '//root//compare_lightweight//saved_models' + '//' + 'model_{epoch:03d}-{val_acc:.4f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True,
                                     verbose=1, save_weights_only=True, period=1)
        # callbacks_list = [checkpoint]
        cosine_lr = CosineAnnealingScheduler(T_max=100, eta_max=0.0001, eta_min=1e-6)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=30, min_lr=1e-6)
        csvlog = CSVLogger('/root/compare_lightweight/saved_models/log.csv')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            # steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            callbacks=[checkpoint, csvlog, earlystopping],
                            )
        path = print_all_file_path('//root//compare_lightweight//saved_models//')
        model.load_weights(path)

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                # rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                # shear_range=0.2,
                # zoom_range=0.2,
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(x_train)
        training_generator=MixupGenerator(x_train,y_train,batch_size=config.batch_size,
                                          alpha=0.4,datagen=None)()
        # callback
        # filepath="saved_models/callback-save-{epoch:02d}-{val_acc:.2f}.hdf5"
        # filepath="saved_models/best_model.hdf5"
        filepath = '//root//compare_lightweight//saved_models'+'//'+'model_{epoch:03d}-{val_accuracy:.4f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True,
                                     verbose=1, save_weights_only=True, period=1)
        # callbacks_list = [checkpoint]
        cosine_lr = CosineAnnealingScheduler(T_max=100, eta_max=0.0001, eta_min=1e-6)
        reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10,min_lr=1e-6)
        csvlog = CSVLogger('/root/compare_lightweight/saved_models/log.csv')
        earlystopping = EarlyStopping(monitor='val_accuracy', patience=10,mode='max')
        # Fit the model on the batches generated by datagen.flow().
        # history = model.fit_generator(datagen.flow(x_train, y_train,
        #     batch_size=batch_size),
        #     steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
        #     epochs=epochs,
        #     validation_data=(x_valid, y_valid),
        #     shuffle=True,
        #     callbacks=[checkpoint,earlystopping,csvlog,cosine_lr],
        #     verbose=2)
        start_t=time.time()
        history = model.fit_generator(generator=training_generator,
                                      steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                                      epochs=epochs,
                                      validation_data=(x_test, y_test),
                                      shuffle=True,
                                      callbacks=[checkpoint, earlystopping, csvlog, cosine_lr],
                                      verbose=2)
        end_t = time.time()
        cost_t = 1000. * (end_t - start_t)
        print("===>success processing img,the average inference time for each image is %.2f ms" % (cost_t / 4172.))
        path=print_all_file_path('//root//compare_lightweight//saved_models//')
        model.load_weights(path)
    return model, history

def evaluate(model, x_test, y_test):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
    return scores

def predict(model, x_test):
    test_classes = model.predict(x_test, verbose=0)
    test_classes = np.argmax(test_classes, axis=1)
    # print(test_classes.shape)
    return test_classes

def print_all_file_path(init_file_path):
    b = 0
    list = os.listdir(init_file_path)
    file = ''
    for i in list:
        if i == 'log.csv':
            continue
        else:
            if int(i[6:9]) > b:
                file = i
            b = int(file[6:9])
    return init_file_path+file

def Modified_mobilenet():
    modified_mobilenet=Modified_MobileNet((224,224,3), n_class=config.class_num, alpha=1.0, include_top=True).build()
    return modified_mobilenet


if __name__ == "__main__":
    print("Hello world")
