from resnet import *
import tensorflow as tf
import PIL.Image as Image
import random

learning_rate=0.00001#学习率
batch_size=16 #批次
max_epochs=500 #epochs次数
max_steps=500
num_class=17 #分类类别数
images = tf.placeholder(dtype=tf.float32,shape=(None, 224, 224, 3))#输入占位符
lables = tf.placeholder(dtype=tf.float32,shape=(None, num_class))#输入占位符
is_training = tf.placeholder(dtype=tf.bool)#输入占位符
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]#rgb通道均值
training=False#True

#读取图像数据
def load_data(data_dir):
    num=0
    train_data=[]
    train_lables=[]
    val_data=[]
    val_lables=[]
    with tf.Session() as sess:
        for line in open(data_dir+'files.txt', mode='r'):
            image_name = str(data_dir) + '/' + str(line.split('\n')[0])#图像路径
            img = Image.open(image_name).resize((224,224),Image.ANTIALIAS)#读取图像，resize
            img_data = np.array(img)#图像转换格式
            #img_data1 = img_data - IMAGENET_MEAN_BGR
            red, green, blue = np.split(img_data, 3,axis=2)#通道分割
            bgr = np.concatenate([blue, green, red], axis=2)
            img_data1 =bgr - IMAGENET_MEAN_BGR#减去均值
            img_data=img_data1
            lable=num/80#计算标签
            temp=num%80#分割训练数据与测试数据
            if temp>=72:
                val_data.append(img_data)
                lable=sess.run(tf.one_hot(lable, num_class))
                val_lables.append(lable)
            else:
                train_data.append(img_data)
                lable = sess.run(tf.one_hot(lable, num_class))
                train_lables.append(lable)
            num=num+1
    cc = list(zip(train_data, train_lables))#打乱位置
    random.shuffle(cc)
    train_data[:], train_lables[:] = zip(*cc)
    cc = list(zip(val_data, val_lables))
    random.shuffle(cc)
    val_data[:], val_lables[:] = zip(*cc)
    return train_data,train_lables,val_data,val_lables

#数据读取
train_data,train_lables,val_data,val_lables=load_data(data_dir='./17flowers/jpg/')
data={}



with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=1)))) as sess:
    saver = add.finturn()#微调 加载权重
    init_weight(data,saver, num_class)
    m_saver = tf.train.Saver()#保存训练好的模型
    # 构造计算图
    logits = inference(images,
                       num_classes=num_class,
                       is_training=is_training,
                       bottleneck=True,
                       sess=sess,
                       data=data)
    #计算损失
    _loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=lables))
    #优化
    train = tf.train.AdamOptimizer(learning_rate).minimize(_loss)
    #计算准确率
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(lables, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    #初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)
    if training:
        #利用已有模型初始化变量
        get_weight(sess, saver, data)
        #计算最大迭代step
        max_steps=len(train_data)/batch_size
        for epoch in range(max_epochs):
            for step in range(max_steps):
                #一个批次数据
                one_batch = np.array(train_data[step * batch_size:(step + 1) * batch_size])
                one_lable = np.array(train_lables[step * batch_size:(step + 1) * batch_size])
                feed_dict = {images: one_batch, lables: one_lable,is_training:training}
                #计算（训练）
                sess.run(train, feed_dict=feed_dict)
                #计算损失与网络输出
                los, p = sess.run([_loss, logits], feed_dict=feed_dict)
                print "epoch=", epoch, "step=", step, "loss=", los#, "accuracy=", acc
            #测试
            if epoch % 10 == 0:
                #epoch = epoch + 1
                len_data = len(val_data)
                sum = 0
                acc = 0
                for i in range(len_data):
                    feed_dict = {images: np.array(val_data)[i].reshape(1, 224, 224, 3),
                                 lables: np.array(val_lables)[i].reshape(1, num_class),
                                    is_training:False}
                    #计算准确率和网络输出
                    acc, pro = sess.run([accuracy, logits], feed_dict=feed_dict)
                    #print pro[0], val_lables[i]
                    #答应结果
                    print np.argmax(pro[0]), np.argmax(val_lables[i])
                    sum = sum + acc
                acc = sum / len_data
                print "test   epoch=", epoch, "step=", step, "loss=", los, "accuracy=", acc
            if epoch % 100 == 0 and epoch!=0:
                #保存训练模型
                m_saver.save(sess, './model/model', global_step=epoch)

    else:
        #flower={0:'Buttercup',1:'',2:,3:,4:,5:,6:,7:,8:,9:,10:,11:,12:,13:,
        #14:,15:,16:}
        '''
        m_saver.restore(sess, './model/model-500')
        len_data = len(val_data)
        sum = 0
        acc = 0
        for i in range(len_data):
            feed_dict = {images: np.array(val_data)[i].reshape(1, 224, 224, 3),
                         lables: np.array(val_lables)[i].reshape(1, num_class),
                         is_training: training}
            acc, pro = sess.run([accuracy, logits], feed_dict=feed_dict)
            print pro[0]
            print np.argmax(pro[0]), np.argmax(val_lables[i])
            sum = sum + acc
        acc = sum / len_data
        print "test accuracy=", acc
        '''
        #加载权重
        m_saver.restore(sess, './model/model-500')
        #读图
        image_name='./17flowers/jpg/image_0001.jpg'
        img = Image.open(image_name).resize((224, 224), Image.ANTIALIAS)
        img_data = np.array(img)
        # img_data1 = img_data - IMAGENET_MEAN_BGR
        red, green, blue = np.split(img_data, 3, axis=2)
        bgr = np.concatenate([blue, green, red], axis=2)
        img_data1 = bgr - IMAGENET_MEAN_BGR
        img_data = img_data1
        feed_dict = {images: np.array(img_data).reshape(1, 224, 224, 3),
                     is_training: training}
        #计算结果
        pro = sess.run(logits, feed_dict=feed_dict)
        #打映
        print pro[0]
        print np.argmax(pro[0])
