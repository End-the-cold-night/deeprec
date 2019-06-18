# -*-coding:utf-8 -*-
import tensorflow as tf
def ESMM(ctr_logits, cvr_logits, ctr_label, cvr_label):
    """
    #param ctr_logits:you can self define ctr model to get ctr_logits
    #param cvr_logits:you can self define cvr model to get cvr_logits
    #param ctr_label:
    #param cvr_label:
    #param params:
    """

    ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
    ctcvr_logits = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
    
    cvr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=cvr_label, logits=ctcvr_logits), name="cvr_loss")
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits), name="ctr_loss")
    loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")

    ctr_accuracy = tf.metrics.accuracy(labels=ctr_label, predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
    cvr_accuracy = tf.metrics.accuracy(labels=cvr_label, predictions=tf.to_float(tf.greater_equal(ctcvr_logits, 0.5)))
    ctr_auc = tf.metrics.auc(ctr_label, ctr_predictions)
    cvr_auc = tf.metrics.auc(cvr_label, ctcvr_logits)
    metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
    tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
    tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('cvr_auc', cvr_auc[1])

    return loss


joint_optimization
