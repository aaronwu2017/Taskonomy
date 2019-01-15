import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model,save_model,load_model
from keras import backend as K
from tqdm import tqdm
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import Callback
import glob as gb
import cv2

import os



def encoder_(inputs):
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


def get_encodes(output):
    Z_train = []

    N_train = X.shape[0] // batch_size
    if N_train * batch_size != X.shape[0]:
        N_train += 1

    i = 0
    for _ in tqdm(range(N_train)):
        Z_train.append(
            output([X[i:min([i + batch_size, X.shape[0]]), :, :, :], 0])[0])
        i += batch_size

    return np.concatenate(Z_train)


def load_data(path,num=20):

    print('Loading data...')

    colors={'black':0,'blue':1,'red':2}
    classes={'jeans':0,'shoes':1,'dress':2,'shirt':3}

    X = []
    Y = []
    L = []

    parents = sorted(os.listdir(path))
    for parent in tqdm(parents):
        s=parent.split('_')
        color_label=colors[s[0]]
        class_label=classes[s[1]]

        img_path = gb.glob(os.path.join(path,parent,'*.jpeg'))
        img_path.extend(gb.glob(os.path.join(path, parent, '*.jpg')))
        img_path=sorted(img_path)
        for i,p in enumerate(img_path):
            L.append(p.split('/')[-2]+'/'+p.split('/')[-1].split('.')[0])
            img=cv2.imread(p)
            img=cv2.resize(img,dsize=(256,256),interpolation=cv2.INTER_AREA)
            X.append(img)
            Y.append([color_label,class_label])

            if i== num-1:
                break


    X=np.array(X).astype('float')/255
    Y=np.array(Y).astype('int')


    return X,Y,L



def test_resnet_model():
    print('[Test]\t[{}]'.format('resnet model'))
    input = Input(shape=(256, 256, 3))
    encoder = encoder_(inputs=input)
    Z= get_encodes(encoder)

    model = load_model('./resnet_model/best_model.h5')

    outputs=model.predict(Z)

    color_losses=-np.log(np.exp(outputs[0])/np.sum(np.exp(outputs[0]),1,keepdims=True))
    category_losses=-np.log(np.exp(outputs[1])/np.sum(np.exp(outputs[1]),1,keepdims=True))

    color_pre_ind=np.argmin(color_losses,1)
    category_pre_ind=np.argmin(category_losses,1)

    with open('./resnet_model/loss.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['','color_loss','category_loss','total_loss','predict'])
        for i in tqdm(range(outputs[0].shape[0])):
            if i>0 and i%num==0:
                writer.writerow('')

            writer.writerow([L[i],color_losses[i,Y[i,0]],category_losses[i,Y[i,1]],
                             color_losses[i, Y[i, 0]]+category_losses[i, Y[i, 1]],
                             colors[color_pre_ind[i]]+'_'+classes[category_pre_ind[i]]])



def load_taskonmy_data(path,type,num=20):

    print('Loading data...')

    colors={'black':0,'blue':1,'red':2}
    classes={'jeans':0,'shoes':1,'dress':2,'shirt':3}

    Z = []
    Y = []
    L=[]

    parents = sorted(os.listdir(path))
    for parent in tqdm(parents):
        s=parent.split('_')
        color_label=colors[s[0]]
        class_label=classes[s[1]]

        img_path = gb.glob(os.path.join(path,parent,type,'*.npy'))
        img_path=sorted(img_path)
        for i,p in enumerate(img_path):
            L.append(p.split('/')[-3] + '/' + p.split('/')[-1].split('.')[0])

            img=np.load(p)
            Z.append(img)
            Y.append([color_label,class_label])

            if i== num-1:
                break


    Z=np.array(Z)
    Y=np.array(Y).astype('int')


    return Z,Y,L






def test_taskonmy_model(type):
    print('[Test]\t[{}]\t[{}]'.format('taskonmy model', type))
    model = load_model('./taskonmy_model/best_model_{}.h5'.format(type))

    outputs=model.predict(Z)

    color_losses=-np.log(np.exp(outputs[0])/np.sum(np.exp(outputs[0]),1,keepdims=True))
    category_losses=-np.log(np.exp(outputs[1])/np.sum(np.exp(outputs[1]),1,keepdims=True))

    color_pre_ind=np.argmin(color_losses,1)
    category_pre_ind=np.argmin(category_losses,1)

    with open('./taskonmy_model/loss_{}.csv'.format(type), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['','color_loss','category_loss','total_loss','predict'])
        for i in tqdm(range(outputs[0].shape[0])):
            if i>0 and i%num==0:
                writer.writerow('')

            writer.writerow([L[i],color_losses[i,Y[i,0]],category_losses[i,Y[i,1]],
                             color_losses[i, Y[i, 0]]+category_losses[i, Y[i, 1]],
                             colors[color_pre_ind[i]]+'_'+classes[category_pre_ind[i]]])




def load_flowers_data(path,type,num=20):

    print('Loading data...')

    Z = []
    Y = []
    L=[]

    img_path = sorted(gb.glob(os.path.join(path, type, '*.npy')))
    for i, p in enumerate(img_path):

        if i%80 > num -1:
            continue


        L.append(str(i // 80) + ':' + p.split('/')[-1].split('.')[0])
        img = np.load(p)
        Z.append(img)
        Y.append([i // 80])



    Z = np.array(Z)
    Y = np.array(Y).astype('int')


    return Z,Y,L



def test_flower_model(type):
    print('[Test]\t[{}]\t[{}]'.format('flower model',type))
    model = load_model('./flower_model/best_model_{}.h5'.format(type))

    outputs=model.predict(Z)

    category_losses=-np.log(np.exp(outputs)/np.sum(np.exp(outputs),1,keepdims=True))

    category_pre_ind=np.argmin(category_losses,1)

    with open('./flower_model/loss_{}.csv'.format(type), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['','loss','predict'])
        for i in tqdm(range(outputs.shape[0])):
            if i>0 and i%num==0:
                writer.writerow('')

            writer.writerow([L[i],category_losses[i,Y[i]],category_pre_ind[i]])




batch_size=32
num=50#测试每个类的前num张图片
colors={0:'black',1:'blue',2:'red'}
classes={0:'jeans',1:'shoes',2:'dress',3:'shirt'}



# X,Y,L=load_data('./dataset',num=num)
# test_resnet_model()
#
#
#
# for type in ['autoencoder','class_1000','class_places','curvature']:
#     Z,Y,L=load_taskonmy_data('./dataset_taskonmy',type=type,num=num)
#     test_taskonmy_model(type=type)



for type in ['autoencoder','class_1000','class_places','curvature']:
    Z,Y,L=load_flowers_data('./flowers',type=type,num=num)
    test_flower_model(type=type)


