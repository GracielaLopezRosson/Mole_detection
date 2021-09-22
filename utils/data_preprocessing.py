from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from sklearn.model_selection import train_test_split


def split_in_train_val_test(X, target):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        target,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    return X_train, X_val, y_train, y_val, X_test, y_test


def get_preprocessed_images(images_directory: str, image_size: tuple,limit:int) -> np.array:
    images = []
    i = 0
    for img in os.listdir(images_directory):
        img = image.load_img(images_directory+img, target_size=image_size)
        img = image.img_to_array(img)

        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        images.append(img)
        i += 1
        if i > limit:
            return np.vstack(images)
    return np.vstack(images)
