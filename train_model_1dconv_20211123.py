# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 00:37:24 2019

@author: tc
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import datetime
import time, os

#取得當日日期
datetime = datetime.datetime.now()
date = "%s_%s" % ("{:0>2d}".format(datetime.month),"{:0>2d}".format(datetime.day))

#用於取得每層超參數數量及使用之記憶體
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True) 


#%%--------------------------------------------------------------
#將當前路徑轉換到放置train和test資料的路徑位置，並取得這個位置存入變數內
os.chdir(r'D:\交大研究所\人工智能競賽\result_program\result_program')
cwd = os.getcwd()+'/'
# model存檔路徑
LOGDIR = './%s_log_BN_model_v2'%date
#訓練及測試集list
training_file_list =[cwd + 'train/'+f for f in os.listdir(cwd + 'train/')]
testing_file_list = [cwd + 'test/'+f for f in os.listdir(cwd + 'test/')]
# 訓練參數設定
num_epochs = 10
learning_rate = 0.01
Batch_size = 500
use_Batch_norm = True

#%%------------------------------------------------------------
ts = time.time()
# 初始化tensorflow graph
tf.reset_default_graph()
# input data placeholder 
x = tf.placeholder(tf.float32, shape=[None, 150,1], name="x")
# input label data placeholder
y = tf.placeholder(tf.float32, shape=[None, 8], name="labels")
# 是否使用batch normalization 的placeholder 
tf_is_training = tf.placeholder(tf.bool, name = "tf_is_training")
# 訓練集及測試集list 的placeholder 
filenames = tf.placeholder(tf.string, shape=[None], name = "filenames")

# 第一層 1dconv layer with BN layer  
conv1 = tf.layers.conv1d(x,filters = 50,kernel_size = 3,strides=1,activation=tf.nn.relu,name = 'conv1')
BN_1 = tf.layers.batch_normalization(conv1, training = tf_is_training,name = 'conv_BN1')

# 第二層 1dconv layer with BN layer 
conv2 = tf.layers.conv1d(BN_1,filters = 50,kernel_size = 3,strides=1,activation=tf.nn.relu,name = 'conv2')
BN_2 = tf.layers.batch_normalization(conv2, training = tf_is_training,name = 'conv_BN2')

# max pooling layer  window size: 5 stride: 5
pool1 = tf.layers.max_pooling1d(BN_2,5,5,name = 'pool1')

# 第三層 1dconv layer with BN layer 
conv3 = tf.layers.conv1d(pool1,filters = 60,kernel_size = 3,strides=1,activation=tf.nn.relu,name = 'conv3')
BN_3 = tf.layers.batch_normalization(conv3, training = tf_is_training,name = 'conv_BN3')

# 第四層 1dconv layer with BN layer 
conv4 = tf.layers.conv1d(BN_3,filters = 60,kernel_size = 3,strides=1,activation=tf.nn.relu,name = 'conv4')
BN_4 = tf.layers.batch_normalization(conv4, training = tf_is_training,name = 'conv_BN4')

# 平面化層 
flat = tf.layers.flatten(BN_4,name='flat')

# 第一層 dense layer node:512
fc3 = tf.layers.dense(flat, units = 256, activation = tf.nn.relu, name= "fc3")
tf.summary.histogram("fc3/relu3", fc3)
BN3 = tf.layers.batch_normalization(fc3, training = tf_is_training, name = "D_BN3")

# 第二層 dense layer node:512
fc4 = tf.layers.dense(BN3, units = 256, activation = tf.nn.relu, name= "fc4")
tf.summary.histogram("fc4/relu4", fc4)
BN4 = tf.layers.batch_normalization(fc4, training = tf_is_training, name = "D_BN4")

# 輸出層 node:8 
pred = tf.layers.dense(BN4, units = 8, activation = tf.nn.relu, name= "pred")
tf.add_to_collection('pred',pred)
tf.summary.histogram("pred", pred)

# 定義loss函數 使用 cross entropy
with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred, labels=y), name="xent")
    tf.add_to_collection('xent',xent) 
    tf.summary.scalar("xent", xent)

# 定義訓練優化器 使用 Adam
with tf.name_scope("train"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_ = tf.train.AdamOptimizer(learning_rate).minimize(xent)
    train_op = tf.group([train_, update_ops])

# 定義準確率acc
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection('accuracy',accuracy)
    tf.summary.scalar("accuracy", accuracy)
# 將所有summary operation集合成summ
summ = tf.summary.merge_all()

# 用於储存模型參數
saver = tf.train.Saver()

# 輸出模型超參數資訊
model_summary()

# 創建一個計算流程
sess = tf.Session()
# 初始化超參數
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# 定義tensorflow寫入器 並將計算圖寫入
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

#--------------------------------tfrecord input pipeline--------------------------------
num_class = 8
# 定義解碼器
def parser(record):
      keys_to_features = {
        "T_curve_string": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                         }
      parsed = tf.parse_single_example(record, keys_to_features)
    
    # Parse the string into an array of temperature corresponding to the dataset 
      T_curve = tf.decode_raw(parsed["T_curve_string"],tf.float32)
      T_curve = tf.reshape(T_curve,[150,1])
      labels = tf.cast(parsed['label'], tf.int32)
      labels = tf.one_hot(labels, num_class)
        
      return T_curve, labels
  
# 創造一個TF dataset
dataset = tf.data.TFRecordDataset(filenames,compression_type = 'GZIP')
# 將 record 解碼成 tensors
dataset = dataset.map(parser) 
# 設定shuffle set的大小 及 Batch size 
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(Batch_size)
# 定義初始化器 可重複初始化不同的tfrecord
iterator = dataset.make_initializable_iterator()
#定義 batch 可持續從dataset中提取資料
Batch = iterator.get_next()
#--------------------------------訓練步驟---------------------------------------
for epoch in range (num_epochs):
#    獲取當前時間
      d = time.asctime( time.localtime(time.time()) )
      print('start train %d at %s' %(epoch+1,d))
      
      with open('%s-train_infor.txt' %date,'a',encoding='utf-8')as f:
          f.write('start train %d at %s \n' %(epoch+1,d))
          
#     每個epoch 訓練開始的時間點
      train_times = time.time()
      
#     初始化 訓練dataset 
      print('initialization')
      sess.run(iterator.initializer,feed_dict = {filenames: training_file_list})
      
      with open('%s-train_infor.txt' %date,'a',encoding='utf-8')as f:
          f.write('initialize complete\n')
          
# 紀錄訓練過的資料數量
      train_data_amount = 0
      
#     重複從dataset提取資料
      while True:
          try:                
#           獲取data and label tensors 
              train_batch = sess.run(Batch)
#           紀錄本次訓練資料數量 
              train_data_amount += train_batch[0].shape[0]
#           執行訓練
              sess.run(train_op, feed_dict={x: train_batch[0], y: train_batch[1], tf_is_training: use_Batch_norm})
        
#         若已用完dataset內的資料
          except:
#             儲存model 
              saver.save(sess, os.path.join(LOGDIR, "BN_model_fold.ckpt" ), epoch+1)
#             輸出完成訓練訊息   
              print('------------ %d epoch done --------------'%(epoch+1))
              
              with open('%s-train_infor.txt' %date,'a',encoding='utf-8')as f:
                  f.write('------------%d epoch done --------------\n'%(epoch+1))
                  
              [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: train_batch[0], y: train_batch[1], tf_is_training: False})
              train_loss = sess.run(xent,feed_dict={x: train_batch[0], y: train_batch[1], tf_is_training: False})
#             寫入 summary   
              writer.add_summary(s, epoch+1)
#             完成訓練的時間點 
              train_timee = time.time()
#             輸出本次epoch之最後一次的acc、loss及訓練時間 
              train_cost_time = (train_timee - train_times) /60
              print('train %d data cost %f mins | epoch: %d |train acc: %f |train loss: %f \n '
                      %(train_data_amount,train_cost_time,epoch+1,train_accuracy,train_loss))
              
              with open('%s-train_infor.txt' %date,'a',encoding='utf-8')as f:
                  f.write('train %d data cost %f mins | epoch: %d |train acc: %f |train loss: %f  \n'
                      %(train_data_amount,train_cost_time,epoch+1,train_accuracy,train_loss))

#             執行驗證  
              with sess.as_default():
#                 初始化驗證dataset
                  sess.run(iterator.initializer,feed_dict={filenames: testing_file_list})
                  print('----------- validate the %dth data ---------------' %(epoch+1))
                  
                  with open('%s-train_infor.txt' %date,'a',encoding='utf-8')as f:
                      f.write('------------ validate the %dth data --------------\n'%(epoch+1))
#                 批次acc及loss暫存區 
                  temp_acc = []
                  temp_loss = []
                  
                  while True:
#                     若尚未用完dataset的資料 
                      try:
#                         獲取驗證batch 
                          validate_batch = sess.run(Batch)
#                         獲取acc及loss 
                          acc= accuracy.eval(feed_dict={x:validate_batch[0], y:validate_batch[1], tf_is_training: False})
                          loss = xent.eval(feed_dict={x:validate_batch[0], y:validate_batch[1], tf_is_training: False})
#                         存入暫存區 
                          temp_acc.append(acc)
                          temp_loss.append(loss)
#                     若用完dataset的資料     
                      except:
#                         計算平均acc及loss 
                          temp_acc = np.array(temp_acc)
                          temp_loss = np.array(temp_loss)
                          valid_acc = temp_acc.mean(axis=0)
                          valid_loss = temp_loss.mean(axis = 0)
#                         獲取驗證結束時間 
                          d = time.asctime( time.localtime(time.time()) )
                          print('complete %dth validate at %s | valid_acc: %f | valid_loss: %f' %((epoch+1),d,valid_acc,valid_loss))
                          
                          with open('%s-train_infor.txt' %date,'a',encoding='utf-8')as f:
                              f.write('complete %dth validate at %s | valid_acc: %f | valid_loss: %f\n' %((epoch+1),d,valid_acc,valid_loss))
                          break
              break
#完成所有訓練的時間點
te = time.time()
#完成訓練的總時間
train_time = (te-ts)/3600
#輸出訓練總時間
print('complete training in %f hr ' %train_time)
with open('%s-train_infor.txt' %date,'a',encoding='utf-8')as f:
    f.write('complete training in %f hr' %train_time)


