import tensorflow as tf
import pickle
import numpy as np
#with tf.Session() as sess:
  #new_saver = tf.train.import_meta_graph('./aaa.ckpt-800.meta')
  #new_saver.restore(sess, 'checkpoint')
  # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
#  y = tf.get_collection('pred_network')[0]

 # graph = tf.get_default_graph()

  # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
  #input_x = graph.get_operation_by_name('input_x').outputs[0]
  #keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

  # 使用y进行预测  
  #sess.run(y, feed_dict={input_x:'我喜欢你',  dropout_keep_prob:1.0})
#saver = tf.train.import_meta_graph("./aaa.ckpt-800.meta")
#with tf.Session() as sess:
 #   saver.restore(sess, "./aaa.ckpt")
#    print(sess.run(tf.get_default_graph().get_tensor_by_name("")))
#W = tf.Variable(tf.truncated_normal([3,128,1,128], stddev=0.1), name='W')
#W = tf.Variable(np.arange(1152).reshape((384, 3)), dtype=tf.float32, name="weights")
cc=np.array([[565, 598, 413, 414, 599, 600, 601, 602, 13, 603]])
dd = np.array([[1,2,3,4,5,6,6,7,8,9]])

#ff = np.fromstring(line, dtype=int, sep=',')
#ff = np.array(list(line))
pkl_file = open('ceshi.pkl','rb')
ff = pickle.load(pkl_file)
print(np.array(ff))
pkl_file.close()

#b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
saver=tf.train.import_meta_graph("./aaa.ckpt-800.meta") 
with tf.Session() as sess:
    saver.restore(sess,'./aaa.ckpt-800')
    #print("weights",sess.run('conv-maxpool-3/W:0'))
    #print("weights",sess.run('output/predictions:0'))
    print(sess.run('output/predictions:0', feed_dict={'input_x:0':ff,  'dropout_keep_prob:0':1.0}))
    #print("weights",sess.run('conv-maxpool-5/W:0'))  
    #graph=tf.get_default_graph()
    #print(sess.run(graph.get_tensor_by_name('global_step:0')))
#    prob_op=graph.get_operation_by_name('predictions')
 #   prediction = graph.get_tensor_by_name('predictiona:0')


  #  print(sess.run(prediction, feed_dict='我喜欢你'))
