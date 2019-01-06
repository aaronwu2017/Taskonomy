import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_data import load_flower_data
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model,save_model,load_model
from keras import backend as K
from tqdm import tqdm
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import Callback

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'


class Save_best(Callback):
    def __init__(self):
        self.acc_cate=0
        super(Save_best,self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        cate_acc=logs.get('val_acc')

        if cate_acc>self.acc_cate:
            save_model(decoder,filepath='./flower_model/best_model_{}.h5'.format(type))
            self.acc_cate=cate_acc
        save_model(decoder,filepath='./flower_model/latest_model_{}.h5'.format(type))




def test_model(path):
    model=load_model(path)
    print('Loading model from {}'.format(path))
    OUT=model.evaluate(Z_test,to_categorical(y_test, num_classes=17),verbose=1,
          batch_size=100)
    print(model.metrics_names)
    print(OUT)




def decoder():
    input = Input(shape=(16, 16, 8))
    output = Conv2D(
    16, kernel_size=(
        3, 3), strides=(
            1, 1), padding='same')(input)
    # None*16*16*16
    output = MaxPooling2D(pool_size=(2, 2))(output)
    # None*8*8*16
    output = Flatten()(output)
    # None*1024
    output = Dense(200, activation='relu')(output)
    # None*200
    output = Dropout(rate=0.5)(output)
    output = Dense(17, activation='softmax')(output)

    return Model(inputs=input, outputs=output)

test_model_=True
type='autoencoder'
model_path='./flower_model/best_model_{}.h5'.format(type)
batch_size = 32
Epochs = 100

Z_train, Z_test, y_train, y_test= load_flower_data('./flowers',type=type,pp=0.2)


decoder = decoder()



if test_model_:
    if os.path.exists(model_path):
        test_model(path=model_path)
    else:
        print('Please train first!')
        raise Exception
else:


    decoder.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), loss='categorical_crossentropy',
        metrics=["accuracy"])
    # train the network to perform multi-output classification
    H = decoder.fit(Z_train,to_categorical(y_train, num_classes=17),
        validation_data=(Z_test,to_categorical(y_test, num_classes=17)),
        epochs=Epochs,
        verbose=1,callbacks=[Save_best()])




    # plot the total loss, category loss, and color loss
    lossNames = ["loss"]
    plt.style.use("ggplot")
    plt.figure(figsize=(13, 13))

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.plot(np.arange(0, Epochs), H.history[l], label=l)
        plt.plot(np.arange(0, Epochs), H.history["val_" + l],
            label="val_" + l)
        plt.legend()

    # save the losses figure
    plt.tight_layout()
    plt.savefig("./flower_model/output_losses_{}.png".format(type))
    plt.close()

    # create a new figure for the accuracies
    accuracyNames = ["acc"]
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 8))

    # loop over the accuracy names
    for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
        plt.title("Accuracy for {}".format(l))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.plot(np.arange(0, Epochs), H.history[l], label=l)
        plt.plot(np.arange(0, Epochs), H.history["val_" + l],
            label="val_" + l)
        plt.legend()

    # save the accuracies figure
    plt.tight_layout()
    plt.savefig("./flower_model/output_accs_{}.png".format(type))
    plt.close()
