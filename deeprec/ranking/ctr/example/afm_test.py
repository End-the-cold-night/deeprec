import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from sklearn.metrics import roc_auc_score
from deeprec.ranking.ctr import data_uploder
from deeprec.ranking.ctr import Afm


def _eval(sess, model):
  y_scores = []
  y_true = []
  for _, uij in data_uploder(test_set, feature_config_dict, test_batch_size):
    y_scores += list(model._eval(sess, uij))
    y_true += list(uij[-1])
  y_scores = np.array(y_scores).flatten()
  y_true = np.array(y_true).flatten()
  test_gauc = roc_auc_score(y_true, y_scores)
  all_result.append(test_gauc)
  print("Now auc is ",  test_gauc)
  global best_auc
  if best_auc < test_gauc:
    best_auc = test_gauc
    model.save(sess, 'save_afm_path/ckpt')
  pickle.dump(all_result, open("auc_afm_all_result.pkl","wb"))
  return None


all_result = []
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)
best_auc = 0
# bs 设置
train_batch_size = 32
test_batch_size = 512



with open('../data/dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  feature_config_dict = pickle.load(f)


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  model = Afm(feature_config_dict)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())  #
  lr = 0.001
  start_time = time.time()
  for _ in range(50):

    random.shuffle(train_set)

    epoch_size = round(len(train_set) / train_batch_size)

    for _, uij in data_uploder(train_set, feature_config_dict, train_batch_size):

      loss = model.train(sess, uij, lr)

      if model.global_step.eval() % 1000 == 0:
          _eval(sess, model)

      if model.global_step.eval() % 336000 == 0:
        lr *= 0.9

    print('Epoch %d DONE\tCost time: %.2f' %
          (model.global_epoch_step.eval(), time.time()-start_time))

    sys.stdout.flush()
    model.global_epoch_step_op.eval()