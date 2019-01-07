import argparse
from keras import backend as K
from keras.models import load_model
#from tensorflow_serving.session_bundle import exporter
from keras.models import model_from_config
from keras.models import Sequential,Model
import tensorflow as tf
import os
import train
from train import Scale
from keras.utils import CustomObjectScope



def convert(prevmodel):

  sess = K.get_session()
   
   # let's convert the model for inference
  K.set_learning_phase(0)  # all new operations will be in test mode from now on
   # serialize the model and get its weights, for quick re-building
  with CustomObjectScope({'Scale': Scale}):
    previous_model = load_model(prevmodel)
    previous_model.summary()

    config = previous_model.get_config()
    weights = previous_model.get_weights()

   # re-build a model where the learning phase is now hard-coded to 0
    try:
      model= Sequential.from_config(config) 
    except:
      model= Model.from_config(config) 
   #model= model_from_config(config)
    model.set_weights(weights)

    print("Input name:")
    print(model.input.name)
    print("Output name:")
    print(model.output.name)
    output_name=model.output.name.split(':')[0]

    #  not sure what this is for
    export_version = 2 # version number (integer)

   #graph_file=export_path+"_graph.pb"
   #ckpt_file=export_path+".ckpt"
   # create a saver 

    saver = tf.train.Saver()
   #tf.train.write_graph(sess.graph_def, '', graph_file)
    save_path = saver.save(sess, '/home/ubuntu/Desktop/resnet_help/taskonomy-master/taskbank/temp/class_1000/model.permanent-ckpt')



convert('/home/ubuntu/Desktop/resnet_help/taskonomy-master/taskbank/tools/saved_models/flowers_ResNet152v2_model.033.h5')
