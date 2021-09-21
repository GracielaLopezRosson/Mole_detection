import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
# import PIL
from sklearn.metrics import classification_report
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from utils import data_preprocessing as dp, modeling as md, plotting

if __name__ == '__main__':
    images_paths = ["data/HAM10000_images_part_1/"]
    image_size = (224, 224)
    limit = 500
    starting_row_part_two = 5000

    metadata_labels = pd.read_csv('csv_files/metadata_labels.csv')

    images_array_lst = []
    for path in images_paths:
        images_array = dp.get_preprocessed_images(path, image_size, limit)
        images_array_lst.append(images_array)
    skin_images = np.vstack(images_array_lst)

    X = skin_images
    target = metadata_labels.dx[:X.shape[0]]

    target_one_hot = tf.keras.utils.to_categorical(target, num_classes=7, dtype="int")

    X_train, X_val, y_train, y_val, X_test, y_test = dp.split_in_train_val_test(X, target_one_hot)


    # train model if model does not exist
    try:
        model = load_model('model/model.h5')
        history_hist = pickle.load(open('model/history', 'rb'))
    except:
        history_hist, model = md.instantiate_model(X_train, X_val, y_train, y_val)

    # evaluate
    y_test_one_column = np.argmax(y_test, axis=1)
    results = model.evaluate(X_test, y_test, batch_size=20)
    print(f"Accuracy on test set is {results[1] * 100:.2f}%")
    y_pred = np.argmax(model.predict(X_test), axis=1)


    print(classification_report(y_test_one_column, y_pred,))
    print(tf.math.confusion_matrix(y_test_one_column, y_pred))


    # precision-recall-curve