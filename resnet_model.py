import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_data import load_data
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
        self.acc_colo=0
        super(Save_best,self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        colo_acc=logs.get('val_color_output_acc')
        cate_acc=logs.get('val_category_output_acc')

        if colo_acc>self.acc_colo and cate_acc>self.acc_cate:
            save_model(decoder,filepath='./resnet_model/best_model.h5')
            self.acc_colo=colo_acc
            self.acc_cate=cate_acc
        save_model(decoder,filepath='./resnet_model/latest_model.h5')




def test_model(path):
    model=load_model(path)
    print('Loading model from {}'.format(path))
    OUT=model.evaluate(Z_test,
		{"category_output": to_categorical(y_test[:, 1], num_classes=4), "color_output": to_categorical(y_test[:, 0], num_classes=3)}
                   ,verbose=1,
          batch_size=100)
    print(model.metrics_names)
    print(OUT)



def get_encodes(output):
    Z_train = []
    Z_test = []

    N_train = X_train.shape[0] // batch_size
    if N_train * batch_size != X_train.shape[0]:
        N_train += 1

    i = 0
    for _ in tqdm(range(N_train)):
        Z_train.append(
            output([X_train[i:min([i + batch_size, X_train.shape[0]]), :, :, :], 0])[0])
        i += batch_size

    N_test = X_test.shape[0] // batch_size
    if N_test * batch_size != X_test.shape[0]:
        N_test += 1

    i = 0
    for _ in tqdm(range(N_test)):
        Z_test.append(
            output([X_test[i:min([i + batch_size, X_test.shape[0]]), :, :, :], 0])[0])
        i += batch_size

    return np.concatenate(Z_train), np.concatenate(Z_test)


def encoder(inputs):
    model = ResNet50(include_top=False,
             weights='imagenet',
             input_tensor=inputs,
             pooling='avg')

    get_resnet50_output = K.function([model.layers[0].input, K.learning_phase()],
                                     [K.reshape(model.layers[-1].output, (-1, 16, 16, 8))])

    return get_resnet50_output


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
    output1 = Dense(3, activation='softmax', name="color_output")(output)

    output2 = Dense(4, activation='softmax', name="category_output")(output)

    return Model(inputs=input, outputs=[output1, output2])


losses = {
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 1.0}

test_model_=True
model_path='./resnet_model/best_model.h5'
batch_size = 32
Epochs = 100

X_train, X_test, y_train, y_test, colors, classes = load_data(
    path='./dataset', pp=0.2)


input = Input(shape=(256, 256, 3))
encoder = encoder(inputs=input)
decoder = decoder()


Z_train, Z_test = get_encodes(encoder)


if test_model_:
    if os.path.exists(model_path):
        test_model(path=model_path)
    else:
        print('Please train first!')
        raise Exception
else:


    decoder.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), loss=losses, loss_weights=lossWeights,
        metrics=["accuracy"])
    # train the network to perform multi-output classification
    H = decoder.fit(Z_train,
        {"category_output": to_categorical(
            y_train[:, 1], num_classes=4), "color_output": to_categorical(y_train[:, 0], num_classes=3)},
        validation_data=(Z_test,
            {"category_output": to_categorical(y_test[:, 1], num_classes=4), "color_output": to_categorical(y_test[:, 0], num_classes=3)}),
        epochs=Epochs,
        verbose=1,callbacks=[Save_best()])




    # plot the total loss, category loss, and color loss
    lossNames = ["loss", "category_output_loss", "color_output_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(np.arange(0, Epochs), H.history[l], label=l)
        ax[i].plot(np.arange(0, Epochs), H.history["val_" + l],
            label="val_" + l)
        ax[i].legend()

    # save the losses figure
    plt.tight_layout()
    plt.savefig("./resnet_model/output_losses.png")
    plt.close()

    # create a new figure for the accuracies
    accuracyNames = ["category_output_acc", "color_output_acc"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

    # loop over the accuracy names
    for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
        ax[i].set_title("Accuracy for {}".format(l))
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Accuracy")
        ax[i].plot(np.arange(0, Epochs), H.history[l], label=l)
        ax[i].plot(np.arange(0, Epochs), H.history["val_" + l],
            label="val_" + l)
        ax[i].legend()

    # save the accuracies figure
    plt.tight_layout()
    plt.savefig("./resnet_model/output_accs.png")
    plt.close()
