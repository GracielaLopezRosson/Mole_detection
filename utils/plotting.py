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