import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import jieba
from text_cnn import TextCNN
from tensorflow.contrib import learn
tf.flags.DEFINE_float('dev_sample_percetage', .1, 'Percentage of the training data to use for validation')
tf.flags.DEFINE_string('positive_data_file', '/home/mwh/CNN_text/positive.txt', 'positive')
tf.flags.DEFINE_string('negative_data_file', '/home/mwh/CNN_text/negative.txt', 'negative')
tf.flags.DEFINE_string('zhongxing_data_file', '/home/mwh/CNN_text/zhongxing.txt', 'zhongxing')
tf.flags.DEFINE_string('prediction_data_file', '/home/mwh/CNN_text/prediction.text', 'prediction')

tf.flags.DEFINE_integer('embedding', 128, 'Dimensionality of character embedding')
tf.flags.DEFINE_string('filter_sizes', '3,4,5', 'filter size')
tf.flags.DEFINE_integer('num_filters', 128, 'num of filters')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2')
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer('batch_size', 64, 'batch_size')
tf.flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
tf.flags.DEFINE_integer('evaluate_every', 100, 'evaluate_every')
tf.flags.DEFINE_integer('checkpoint_every', 100, 'saving...')
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
#print('\nParametwea:')
#for attr, values in sorted(FLAGS.__flags.items()):
#    print('{}={}'.format(attr.upper(), values))
x_text, y,i = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file, FLAGS.zhongxing_data_file,FLAGS.prediction_data_file)
max_document_length = max(len(list(jieba.cut(x,cut_all=False))) for x in x_text)
#max_document_length = 10
#print(max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# print(type(vocab_processor))
x = np.array(list(vocab_processor.fit_transform(x_text)))
#print(x)
#print(y)
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
# print(shuffle_indices.size)
x_shuffled = x[:i][shuffle_indices]
y_shuffled = y[shuffle_indices]
ceshi = x[len(x_shuffled):].tolist()

f = open('ceshi.pkl','wb')
pickle.dump(ceshi,f)
f.close()
print(len(x_shuffled))
print(type(x))

#print(y)
dev_sample_index = -1 * int(FLAGS.dev_sample_percetage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#print(len(vocab_processor.vocabulary_))
#print(x_train)
#print(y_train)
#print('Vocabulary Size:{:d}'.format(len(vocab_processor.vocabulary_)))
#print('train/dev_split:{:d}/{:d}'.format(len(y_train), len(y_dev)))
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=y_train.shape[1],
                      vocab_size=len(vocab_processor.vocabulary_),
                      embedding_size=FLAGS.embedding,
                      filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                      num_filters=FLAGS.num_filters,
                      l2_reg_lambda=FLAGS.l2_reg_lambda)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        grad_summaries = []
        for g, v in grads_and_vars:
 #           print(g,v)
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
  #      print("Writing to {}\n".format(out_dir))
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())
        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy]
                ,feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
        # def dev_step(x_batch, y_batch, writer=None):
        #     num = 20
        #     x_batch = x_batch.tolist()
        #     y_batch = y_batch.tolist()
        #     l = len(y_batch)
        #     l_20 = int(l / num)
        #     x_set = []
        #     y_set = []
        #     for i in range(num - 1):
        #         x_temp = x_batch[i * l_20:(i + 1) * l_20]
        #         x_set.append(x_temp)
        #         y_temp = y_batch[i * l_20:(i + 1) * l_20]
        #         y_set.append(y_temp)
        #     x_temp = x_batch[(num - 1) * l_20:]
        #     x_set.append(x_temp)
        #     y_temp = y_batch[(num - 1) * l_20:]
        #     y_set.append(y_temp)
        #
        #     # 每个batch验证集计算一下准确率，num个batch再平均
        #     lis_loss = []
        #     lis_accu = []
        #     for i in range(num):
        #         feed_dict = {
        #             cnn.input_x: np.array(x_set[i]),
        #             cnn.input_y: np.array(y_set[i]),
        #             cnn.dropout_keep_prob: 1.0
        #         }
        #         step, summaries, loss, accuracy = sess.run(
        #             [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        #             feed_dict)
        #         lis_loss.append(loss)
        #         lis_accu.append(accuracy)
        #         time_str = datetime.datetime.now().isoformat()
        #         print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        #         dev_summary_writer.add_summary(summaries, step)
        #     print("test_loss and test_acc" + "\t\t" + str(sum(lis_loss) / num) + "\t\t" + str(sum(lis_accu) / num))
        #     if writer:
        #         writer.add_summary(summaries, step)
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        print(batches)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, './aaa.ckpt', global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
