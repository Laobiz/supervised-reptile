# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools

# Display the Digit from the image
def display_digit(image, label=None, pred_label=None):
    if image.shape == (784,):
        image = image.reshape((28, 28))
    label = np.argmax(label, axis=0)
    if pred_label is None and label is not None:
        plt.title('Label: %d' % (label))
    elif label is not None:
        plt.title('Label: %d, Pred: %d' % (label, pred_label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

# Display the convergence of the error while training
def display_convergence(error):
    plt.plot(error)
    plt.title('Error of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues, NAME="", SAVE=False):
    '''
    Credits to: Yassine Ghouzam
    Url: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if SAVE == True:
        plt.savefig('confusion_matrix'+NAME+'.png')
    else:
        plt.show()

def make_hparam_string(learning_rate, conv_layer, fc_layer, batch_size=256):
    conv_param = "conv_layer=" + str(conv_layer)
    fc_param = "fc_layer=" + str(fc_layer)
    batch_param = "batch_size=" + str(batch_size)
    return "lr_%.0E %s %s %s" % (learning_rate, conv_param, fc_param, batch_param)