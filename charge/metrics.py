import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def masked_softmax_cross_entropy(preds, placeholders):
    """Softmax cross-entropy loss with masking."""
    labels = placeholders['labels']
    mask = placeholders['labels_mask']
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, placeholders):
    """Accuracy with masking."""
    labels = placeholders['labels']
    mask = placeholders['labels_mask']
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))

    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
