import itertools
import numpy as np
import tensorflow
from matplotlib import pyplot as plt


def plot_history(history: tensorflow.keras.callbacks.History):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(history['accuracy'], label='training set')
    axs[0].plot(history['val_accuracy'], label='validation set')
    axs[0].set(xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])

    axs[1].plot(history['loss'], label='training set')
    axs[1].plot(history['val_loss'], label='validation set')
    axs[1].set(xlabel='Epoch', ylabel='Loss', ylim=[0, 20])

    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')
    plt.show()


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues, figsize=(10,10)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2) * 100
    print(cm)
    thresh = cm.max() / 2.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.savefig('./visuals/confusion_matrix.jpg')
    # plt.show()
