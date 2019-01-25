from __future__ import absolute_import, division, print_function

import argparse
import importlib
import itertools
import matplotlib
matplotlib.use('Agg')
import time
from   multiprocessing import Pool
import numpy as np
import os
import pdb
import pickle
import subprocess
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import scipy.misc
from skimage import color
import init_paths
from models.sample_models import *
from lib.data.synset import *
import scipy
import skimage
import skimage.io
import transforms3d
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from task_viz import *
import random
import utils
import models.architectures as architectures
from   data.load_ops import resize_rescale_image
from   data.load_ops import rescale_image
import utils
import lib.data.load_ops as load_ops

parser = argparse.ArgumentParser(description='Viz Single Task')

parser.add_argument('--task', dest='task')
parser.set_defaults(task='NONE')

parser.add_argument('--input', dest='input_path')
parser.set_defaults(im_name='NONE')

parser.add_argument('--output', dest='output_path')
parser.set_defaults(store_name='NONE')

parser.add_argument('--store-rep', dest='store_rep', action='store_true')
parser.set_defaults(store_rep=False)

parser.add_argument('--store-pred', dest='store_pred', action='store_true')
parser.set_defaults(store_pred=False)

parser.add_argument('--on-screen', dest='on_screen', action='store_true')
parser.set_defaults(on_screen=False)

tf.logging.set_verbosity(tf.logging.ERROR)

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization jigsaw \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
list_of_tasks = list_of_tasks.split()

def generate_cfg(task):
    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    CONFIG_DIR = os.path.join(repo_dir, 'experiments/final', task)
    ############## Load Configs ##############
    import utils
    import data.load_ops as load_ops
    from   general_utils import RuntimeDeterminedEnviromentVars
    cfg = utils.load_config( CONFIG_DIR, nopause=True )
    RuntimeDeterminedEnviromentVars.register_dict( cfg )
    cfg['batch_size'] = 1
    if 'batch_size' in cfg['encoder_kwargs']:
        cfg['encoder_kwargs']['batch_size'] = 1
    cfg['model_path'] = os.path.join( repo_dir, 'temp', task, 'model.permanent-ckpt' )
    print(cfg['model_path'])
    cfg['root_dir'] = repo_dir
    return cfg



def run_one_task(model, training_runners, cfg,  img_path, args):
    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars
    img = load_raw_image_center_crop(img_path)
    img = skimage.img_as_float(img)
    scipy.misc.toimage(np.squeeze(img), cmin=0.0, cmax=1.0).save(img_path)
    
    task = args.task
    if task not in list_of_tasks:
        raise ValueError('Task not supported')

    
    if task == 'jigsaw' :
        img = cfg[ 'input_preprocessing_fn' ]( img, target=cfg['target_dict'][random.randint(0,99)], 
                                                **cfg['input_preprocessing_fn_kwargs'] )
    else:
        img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )

    img = img[np.newaxis,:]

    if task == 'class_places' or task == 'class_1000':
        synset = get_synset(task)

  

    predicted, representation = training_runners['sess'].run( 
            [ model.decoder_output,  model.encoder_output ], feed_dict={model.input_images: img} )

    img_name = os.path.basename(img_path)
    s_name, file_extension = os.path.splitext(img_name)
    save_path = "{}/{}.npy".format(args.output_path, s_name)

    with open(save_path, 'wb') as fp:
        np.save(fp, np.squeeze(representation))


    return


def  batch_run_to_task():
    # batch to precess task
    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars
    import glob
    from tqdm import tqdm

    tf.logging.set_verbosity(tf.logging.ERROR)
    args = parser.parse_args()
    print("input path :{}".format(args.input_path))
    task = args.task
    cfg = generate_cfg(task)

    # Since we observe that areas with pixel values closes to either 0 or 1 sometimes overflows, we clip pixels value
    low_sat_tasks = 'autoencoder curvature denoise edge2d edge3d \
    keypoint2d keypoint3d \
    reshade rgb2depth rgb2mist rgb2sfnorm \
    segment25d segment2d room_layout'.split()
    if task in low_sat_tasks:
        cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image_low_sat


    print("Doing {task}".format(task=task))
    general_utils = importlib.reload(general_utils)
    tf.reset_default_graph()
    training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }

     ############## Set Up Inputs ##############
    # tf.logging.set_verbosity( tf.logging.INFO )
    setup_input_fn = utils.setup_input
    inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
    RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
    RuntimeDeterminedEnviromentVars.populate_registered_variables()
    start_time = time.time()
   

    model = utils.setup_model( inputs, cfg, is_training=False )
    m = model[ 'model' ]
    model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
   
 
    #imgs_path = [ os.path.join(args.input_path, name) for name in glob.glob1(args.input_path, "*")]
    root_path = os.path.dirname(args.input_path)
    print("root path : {}".format(root_path))
    imgs_path = []
    with open(args.input_path, 'r') as filt_ptr:
        for line in filt_ptr:
            split_line = line.strip().split(' ')
            imgs_path.append(os.path.join(root_path, split_line[0]))
    print("load {} imags".format(len(imgs_path)))

    if not os.path.isdir(args.output_path):
        print("create path")
        os.makedirs(args.output_path)
    
    for img_path in tqdm(imgs_path):
        #print("img path: {}".format(img_path))
        run_one_task(m, training_runners, cfg, img_path, args)

    
    ############## Clean Up ##############
    training_runners[ 'coord' ].request_stop()
    training_runners[ 'coord' ].join()
    #print("Done: {}".format(config_name))

    ############## Reset graph and paths ##############            
    tf.reset_default_graph()
    training_runners['sess'].close()


if __name__ == '__main__':
    batch_run_to_task()

