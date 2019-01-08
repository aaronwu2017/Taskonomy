import os
import glob as gb
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split




def load_data(path,pp=0.2):

    print('Loading data...')

    colors={'black':0,'blue':1,'red':2}
    classes={'jeans':0,'shoes':1,'dress':2,'shirt':3}

    dataname=path.split('/')[-1]

    if os.path.exists('{}.npz'.format(dataname)):
        D=np.load('{}.npz'.format(dataname))
        X_train=D['arr_0']
        X_test=D['arr_1']
        y_train=D['arr_2']
        y_test=D['arr_3']
    else:
        X = []
        Y = []

        parents = sorted(os.listdir(path))
        for parent in tqdm(parents):
            s=parent.split('_')
            color_label=colors[s[0]]
            class_label=classes[s[1]]

            img_path = gb.glob(os.path.join(path,parent,'*.jpeg'))
            img_path.extend(gb.glob(os.path.join(path, parent, '*.jpg')))
            img_path=sorted(img_path)
            for p in img_path:
                img=cv2.imread(p)
                img=cv2.resize(img,dsize=(256,256),interpolation=cv2.INTER_AREA)
                X.append(img)
                Y.append([color_label,class_label])

        X=np.array(X).astype('float')/255
        Y=np.array(Y).astype('int')

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=pp, random_state=0)
        np.savez('{}.npz'.format(dataname),X_train,X_test,y_train,y_test)

    return X_train, X_test, y_train, y_test, colors, classes




def load_taskonmy_data(path,type,pp=0.2):
    print('Loading data...')

    colors = {'black': 0, 'blue': 1, 'red': 2}
    classes = {'jeans': 0, 'shoes': 1, 'dress': 2, 'shirt': 3}

    dataname = path.split('/')[-1]

    if os.path.exists('{}_{}.npz'.format(dataname,type)):
        D = np.load('{}_{}.npz'.format(dataname,type))
        Z_train = D['arr_0']
        Z_test = D['arr_1']
        y_train = D['arr_2']
        y_test = D['arr_3']
    else:
        Z = []
        Y = []

        parents = sorted(os.listdir(path))
        for parent in tqdm(parents):
            s = parent.split('_')
            color_label = colors[s[0]]
            class_label = classes[s[1]]

            img_path = sorted(gb.glob(os.path.join(path, parent,type, '*.npy')))
            for p in img_path:
                img = np.load(p)
                Z.append(img)
                Y.append([color_label, class_label])

        Y = np.array(Y).astype('int')

        Z_train, Z_test, y_train, y_test = train_test_split(Z, Y, test_size=pp, random_state=0)
        np.savez('{}_{}.npz'.format(dataname,type), Z_train, Z_test, y_train, y_test)

    return Z_train, Z_test, y_train, y_test, colors, classes

def load_flower_data(path,type,pp=0.2):
    print('Loading data...')


    dataname = path.split('/')[-1]

    if os.path.exists('{}_{}.npz'.format(dataname, type)):
        D = np.load('{}_{}.npz'.format(dataname, type))
        Z_train = D['arr_0']
        Z_test = D['arr_1']
        y_train = D['arr_2']
        y_test = D['arr_3']
    else:
        Z = []
        Y = []

        img_path = sorted(gb.glob(os.path.join(path, type, '*.npy')))
        for i,p in enumerate(img_path):
            img = np.load(p)
            Z.append(img)
            Y.append([i//80])

        Y = np.array(Y).astype('int')

        Z_train, Z_test, y_train, y_test = train_test_split(Z, Y, test_size=pp, random_state=0)
        np.savez('{}_{}.npz'.format(dataname, type), Z_train, Z_test, y_train, y_test)

    return Z_train, Z_test, y_train, y_test

# load_taskonmy_data('/home//dataset_taskonmy',type='autoencoder',pp=0.2)
# load_data('/home//dataset',pp=0.2)
# load_flower_data('/home/flowers',type='autoencoder')
