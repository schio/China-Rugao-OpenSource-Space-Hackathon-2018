import pandas as pd
import json
from pprint import pprint as p
import sys,os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import tensorflow as tf
import numpy as np
import csv,random

def dataLoad(filename):
    fp = open(filename, 'r')
    csv_reader = csv.reader(fp)
    btcusdt_data = []
    for row in csv_reader:
        btcusdt_data.append(row)
    
    # 사용하지 않을 데이터 제거
    print(btcusdt_data[0])
    del btcusdt_data[0]

    dataset = []
    for data in btcusdt_data:
        data_point = []
        is_null = False
        for element in data[1:]:
            if element == 'null':
                is_null = True
            else:
                data_point.append(float(element))
        if is_null:
            continue
        dataset.append(np.array(data_point))
    dataset = np.array(dataset)

    # print(dataset)
    # print(np.shape(dataset))
    return dataset
def model(inputs_ph, init_state, seq_len, out_seq_len):
  '''
  arguments:
      inputs_ph: input placeholder
      init_state: 초기 상태 placeholder
      seq_len: input의 time 길이
      out_seq_len: output의 time 길이
  return:
      softmax result
  '''
  with tf.variable_scope('rnn') as scope:
      # encoder
      rc1 = init_state
      for i in range(seq_len):
         input_prev = tf.concat([inputs_ph[i], rc1], 1)
         rc1 = fully_connected_layer(input_prev, NODE_NUM, 'rc1')
         rc1 = tf.nn.tanh(rc1)
         rc1 = tf.nn.tanh(rc1)
         rc1 = tf.nn.tanh(rc1)
         rc1 = tf.nn.tanh(rc1)
         logits = fully_connected_layer(rc1, 5, 'logits')

         if i==0:
            tf.get_variable_scope().reuse_variables()

      # decoder
      logits_list = [tf.reshape(logits, (-1, 1, 5))]
      for i in range(out_seq_len-1):
         input_prev = tf.concat([logits, rc1], 1)
         rc = fully_connected_layer(input_prev, NODE_NUM, 'rc1')
         rc1 = tf.nn.tanh(rc)
         rc2 = tf.nn.tanh(rc1)
         rc3 = tf.nn.tanh(rc2)
         rc4 = tf.nn.tanh(rc3)
         rc5 = tf.nn.tanh(rc4)
         rc6 = tf.nn.tanh(rc5)
         rc7 = tf.nn.tanh(rc6)
         rc8 = tf.nn.tanh(rc7)
         rc9 = tf.nn.tanh(rc8)
         rc10 = tf.nn.tanh(rc9)
         rc11 = tf.nn.tanh(rc10)
         rc12 = tf.nn.tanh(rc11)
         rc13 = tf.nn.tanh(rc12)
         rc14 = tf.nn.tanh(rc13)
         rc15 = tf.nn.tanh(rc14)
         rc16 = tf.nn.tanh(rc15)
         rc17 = tf.nn.tanh(rc16)
         rc18 = tf.nn.tanh(rc17)
         rc19 = tf.nn.tanh(rc18)
         rc20 = tf.nn.tanh(rc19)
         rc21 = tf.nn.tanh(rc20)
         rc22 = tf.nn.tanh(rc21)
         rc23 = tf.nn.tanh(rc22)
         logits = fully_connected_layer(rc1, 5, 'logits')
         logits_list.append(tf.reshape(logits, (-1, 1, 5)))

      result = tf.concat(logits_list, axis=1)

  return result

def fully_connected_layer(input_tensor, out_node_num, name):
   input_dim = input_tensor.shape.as_list()
   W = tf.get_variable(name=name+'_weights', shape=[input_dim[1], out_node_num])
   b = tf.get_variable(name=name+'_biases', shape=[out_node_num])
   result = tf.matmul(input_tensor, W) + b
   return result

def read_data():
   dataset = dataLoad("/home/scio/Desktop/2018_china_hack/preprodata_10.csv")
   dataset=tf.keras.utils.normalize(dataset)
   train_input, train_label, test_input, test_label = timeSlicer(dataset)
   return train_input, train_label, test_input, test_label

def timeSlicer(dataset):
    seg_start = 0
    seg_end = 100 #시계열 사이즈 셋팅 
    train_size=1500 #트레인으로 몇개 쓸 지
    num_of_pred=6 #캔들 몇개 예측할 지 
    inputs = []
    labels = []
    while seg_end + num_of_pred <= len(dataset):
        input_segment = dataset[seg_start:seg_end]
        label_segment = dataset[seg_end:seg_end+num_of_pred]
        inputs.append(input_segment)
        labels.append(label_segment)
        seg_start += 1
        seg_end += 1
    inputs = np.array(inputs)
    labels = np.array(labels)

    train_input = inputs[:train_size]
    train_label = labels[:train_size]

    test_input = inputs[train_size:]
    test_label = labels[train_size:]

    return train_input, train_label, test_input, test_label

def objective_graph(prediction, labels):
   lsm = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - labels), [1, 2]))
   return lsm


NODE_NUM = 512
SEQ_LEN = 100 #시계열 사이즈 셋팅
OUT_SEQ_LEN = 6 #캔들 몇개 예측할지
plt.ion()
fig = plt.figure(1)
train_input, train_label, test_input, test_label = read_data()

inputs = tf.placeholder(tf.float32, [None, SEQ_LEN, 5])

inputs_max = tf.reduce_max(inputs, 1, True)
inputs_min = tf.reduce_min(inputs, 1, True)
normed_inputs = (inputs - inputs_min) / (inputs_max - inputs_min)

inputs_t = tf.transpose(normed_inputs, [1, 0, 2])
inputs_r = tf.reshape(inputs_t, [-1, 5])

inputs_s = tf.split(inputs_r, SEQ_LEN, axis=0)

init_state = tf.placeholder(tf.float32, [None, NODE_NUM])
labels = tf.placeholder(tf.float32, [None, OUT_SEQ_LEN, 5])

normed_labels = (labels - inputs_min) / (inputs_max - inputs_min)

prediction = model(inputs_s, init_state, SEQ_LEN, OUT_SEQ_LEN)
loss = objective_graph(prediction, normed_labels)

train_vars = tf.trainable_variables()
for var in train_vars:
   print(var.name)

# train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
# train_op = tf.train.AdagradOptimizer(1e-4).minimize(loss)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
init_op = tf.global_variables_initializer()

r=20000
lsm_scalar = tf.summary.scalar('25_layer_ie4_r20000_train1500_test_4',loss) #tensorboard
summary=tf.summary.merge_all() #tensorboard
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    with tf.device('/gpu:0'):
        sess.run(init_op)
        writer = tf.summary.FileWriter('/home/scio/Desktop/2018_china_hack/train/100_',sess.graph) #tensorboard
        for step in range(r):

            shuffle_index = np.random.permutation(len(train_input))
            batch_xs = train_input[shuffle_index[:100]]
            batch_ys = train_label[shuffle_index[:100]]
            feed_dict = {}
            feed_dict[inputs] = batch_xs
            feed_dict[labels] = batch_ys

            init_state_value = np.zeros([len(batch_ys), NODE_NUM])
            feed_dict[init_state] = init_state_value
            sess.run(train_op, feed_dict=feed_dict)
            
            input_value = sess.run(normed_inputs, feed_dict=feed_dict)
            label_value = sess.run(normed_labels, feed_dict=feed_dict)
            pred_value = sess.run(prediction, feed_dict=feed_dict)
            loss_value = sess.run(loss, feed_dict=feed_dict)
            if step % (r/10) == 0:
                print("--------------------------------------------------")
                print('%dst step train loss value: ',step, loss_value)
            show_loss = sess.run(lsm_scalar, feed_dict=feed_dict) #tensorboard
            writer.add_summary(show_loss, step) #tensorboard
        # saver.save(sess, '/home/scio/Desktop/2018_china_hack/model/test_0') #0.01941 r_10000, trainsize_1500
        # saver.save(sess, '/home/scio/Desktop/2018_china_hack/model/test_1') #0.01941 r_1000, trainsize_96000
        # saver.save(sess, '/home/scio/Desktop/2018_china_hack/model/test_2') #1.225 r_10000, trainsize_96000
        # saver.save(sess, '/home/scio/Desktop/2018_china_hack/model/test_3') #0.6717 r_60000, trainsize_96000
        saver.save(sess, '/home/scio/Desktop/2018_china_hack/model/test_4')

for i in range(len(test_label)):
	plt.cla()
	plt.plot(range(-99, 1), input_value[i, :, 1], 'b')
	plt.plot(range(1, 7), label_value[i, :, 1], 'g')
	plt.plot(range(1, 7), pred_value[i, :, 1], 'r--')
	plt.title('test')
	path="/home/scio/Desktop/2018_china_hack/pred/test_4/"+"test"+str(i)+".png"
	plt.savefig(path)
		
