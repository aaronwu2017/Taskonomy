import keras
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt

import copy
import numpy as np
import sys
import time
import PIL.Image as Image
import random

class FrozenGraph(object):
  def __init__(self, model, shape):
    shape = (None, shape[0], shape[1], shape[2])
    x_name = 'image_tensor_x'
    with K.get_session() as sess:
        x_tensor = tf.placeholder(tf.float32, shape, x_name)
        K.set_learning_phase(0)
        y_tensor = model(x_tensor)
        y_name = y_tensor.name[:-2]
        graph = sess.graph.as_graph_def()
        graph0 = tf.graph_util.convert_variables_to_constants(sess, graph, [y_name])
        graph1 = tf.graph_util.remove_training_nodes(graph0)

    self.x_name = [x_name]
    self.y_name = [y_name]
    self.frozen = graph1

class TfEngine(object):
  def __init__(self, graph):
    g = tf.Graph()
    with g.as_default():
      x_op, y_op = tf.import_graph_def(
          graph_def=graph.frozen, return_elements=graph.x_name + graph.y_name)
      self.x_tensor = x_op.outputs[0]
      self.y_tensor = y_op.outputs[0]

    config = tf.ConfigProto(gpu_options=
      tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
      allow_growth=True))

    self.sess = tf.Session(graph=g, config=config)

  def infer(self, x):
    y = self.sess.run(self.y_tensor,
      feed_dict={self.x_tensor: x})
    return y

class TftrtEngine(TfEngine):
  def __init__(self, graph, batch_size, precision):
    tftrt_graph = tftrt.create_inference_graph(
      graph.frozen,
      outputs=graph.y_name,
      max_batch_size=batch_size,
      max_workspace_size_bytes=1 << 30,
      precision_mode=precision,
      minimum_segment_size=2)

    opt_graph = copy.deepcopy(graph)
    opt_graph.frozen = tftrt_graph
    super(TftrtEngine, self).__init__(opt_graph)
    self.batch_size = batch_size

  def infer(self, x):
    num_tests = x.shape[0]
    y = np.empty((num_tests, 17), np.float32)
    batch_size = self.batch_size

    for i in range(0, num_tests, batch_size):
      x_part = x[i : i + batch_size]
      y_part = self.sess.run(self.y_tensor,
        feed_dict={self.x_tensor: x_part})
      y[i : i + batch_size] = y_part
    return y

def verify(result, ans):
  num_tests = ans.shape[0]
  error = 0
  for i in range(0, num_tests):
    a = np.argmax(ans[i])
    r = np.argmax(result[i])
    if (a != r) : error += 1

  if (error == 0) : print('PASSED')
  else            : print('FAILURE')



def load_data(data_dir):
    num=0
    train_data=[]
    train_lables=[]
    test_data=[]
    test_lables=[]
    for line in open(data_dir+'files.txt', mode='r'):
        image_name = str(data_dir) + '/' + str(line.split('\n')[0])
        img = Image.open(image_name).resize((224,224),Image.ANTIALIAS)
        img_data = np.array(img)
        lable=int(num/80)
        temp=num%80 
        if temp<5:
            test_data.append(img_data)
            test_lables.append(lable)
        else:
            train_data.append(img_data)
            train_lables.append(lable)
        num=num+1
    cc = list(zip(train_data, train_lables))
    random.shuffle(cc)
    train_data[:], train_lables[:] = zip(*cc)
    cc = list(zip(test_data, test_lables))
    random.shuffle(cc)
    test_data[:], test_lables[:] = zip(*cc)

    x_train = np.array(train_data)
    y_train = np.array(train_lables)
    x_test = np.array(test_data)
    y_test = np.array(test_lables)
    return x_test,y_test





def main():
  # load pre-trained model
  model = load_model("./saved_models/flowers_ResNet29v2_model.056.h5")
  model.summary()

  # load mnist dataset
  x_test,y_test=load_data(data_dir='./17flowers/jpg/')

  batch_size = 1
  img_h = x_test.shape[1]
  img_w = x_test.shape[2]

  # use Keras to do infer
  t0 = time.time()
  y_keras = model.predict(x_test)
  t1 = time.time()
  print('Keras time', t1 - t0)

  frozen_graph = FrozenGraph(model, (img_h, img_w, 3))

  tf_engine = TfEngine(frozen_graph)
  t0 = time.time() 
  y_tf = tf_engine.infer(x_test)
  t1 = time.time()
  print('Tensorflow time', t1 - t0)
  verify(y_tf, y_keras)

  tftrt_engine = TftrtEngine(frozen_graph, batch_size, 'FP32')
  t0 = time.time() 
  y_tftrt = tftrt_engine.infer(x_test)
  t1 = time.time()
  print('TFTRT time', t1 - t0)
  verify(y_tftrt, y_keras)

  tftrt_engine = TftrtEngine(frozen_graph, batch_size, 'INT8')
  t0 = time.time() 
  y_tftrt = tftrt_engine.infer(x_test)
  t1 = time.time()
  print('TFTRT_INT8 time', t1 - t0)
  verify(y_tftrt, y_keras)

if __name__ == "__main__":
  main()
