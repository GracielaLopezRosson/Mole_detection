from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
from sklearn.model_selection import train_test_split
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import cv2
import os




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


def read_images_with_pil(images_directory, image_size, limit):
    images = []
    i = 0
    for img in os.listdir(images_directory):
        img = Image.open(images_directory + img)
        tf_image = np.array(img)
        img_resized = np.resize(tf_image, (224, 224, 3))
        # img_resized = img_resized[:, :, ::-1]  # RGB to BGR
        # img_reshaped = img_resized.reshape((1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
        # img_scaled = img_reshaped / 255
        images.append(img_resized)
        i += 1
        if i > limit:
            return np.vstack(images)
    return np.vstack(images)
# img = Image.open(images_directory+img)

def read_images(images_directory, image_size, limit):
    images = []
    i = 0
    for img in os.listdir(images_directory):
        # img = Image.open(images_directory+img)
        img = image.load_img(images_directory + img, target_size=image_size)
        # print(img.size) # (224,224)
        tf_image = np.array(img)
        img_resized = np.resize(tf_image, (224, 224, 3))
        # print(img_resized.shape)
        img_reshaped = img_resized.reshape((1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
        images.append(img_reshaped)
        i += 1
        if i > limit:
            return np.vstack(images)
            # return images
    return np.vstack(images)
    # return images


def hair_removal(images):
    images_no_hair = []
    for img in images:
        img = remove_hair(img)
        img_reshaped = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

        images_no_hair.append(img_reshaped)
    return np.vstack(images_no_hair)


def preprocess_no_hair_images(images_no_hair):
    images = []
    for img in images_no_hair:
        # img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        print(img)
        images.append(img)
    return np.vstack(images)

def augment_the_data_balance(images_no_hair):
    gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=0.15,
        # zoom_range=0.1,
        # channel_shift_range=10.,
        # horizontal_flip=True)
    )

    augmented_images_list = []
    for idx, img in enumerate(images_no_hair):

        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        aug_iter = gen.flow(img)
        augmented_images_one_image = [next(aug_iter)[0].astype(np.uint8) for i in range(2)]
        augmented_images_list.append((augmented_images_one_image))

    # aug_iter = gen.flow(images_no_hair)
    # augmented_images_one_image = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
    # augmented_images_list.append((augmented_images_one_image))
    return np.vstack(augmented_images_list)

def augment_the_data(images_no_hair):
    gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.1,
        channel_shift_range=10.,
        horizontal_flip=True)


    augmented_images_list = []
    # for idx, img in enumerate(images_no_hair):
    for img in images_no_hair:
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        aug_iter = gen.flow(img)
        augmented_images_one_image = [next(aug_iter)[0].astype(np.uint8) for i in range(2)]
        augmented_images_list.append((augmented_images_one_image))

    # aug_iter = gen.flow(images_no_hair)
    # augmented_images_one_image = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
    # augmented_images_list.append((augmented_images_one_image))
    return np.vstack(augmented_images_list)


def get_preprocessed_images_transfer_learning(images_directory: str, image_size: tuple, limit:int) -> np.array:
    images = []
    i = 0
    for img in os.listdir(images_directory):
        img = image.load_img(images_directory+img, target_size=image_size)
        img = image.img_to_array(img)
        # img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        images.append(img)
        i += 1
        if i > limit:
            return np.vstack(images)
    return np.vstack(images)


def get_preprocessed_images_no_tl(images_directory: str, limit:int) -> np.array:
    images = []
    i = 0
    for img in os.listdir(images_directory):
        img = Image.open(images_directory+img)
        tf_image = np.array(img)
        img_resized = np.resize(tf_image, (224, 224, 3))
        img_resized = img_resized[:, :, ::-1]  # RGB to BGR
        img_reshaped = img_resized.reshape((1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
        img_scaled = img_reshaped / 255
        images.append(img_scaled)
        i += 1
        if i > limit:
            return np.vstack(images)
    return np.vstack(images)


# def prepare_one_image_no_tl(img):
#     tf_image = np.array(img)
#     img_resized = np.resize(tf_image, (224, 224, 3))
    # st.write(img_resized.shape)
    # img_resized = img_resized[:, :, ::-1]  # RGB to BGR
    # img_reshaped = img_resized.reshape(
    #     (1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
    # st.write(img_reshaped.shape)
    # img_scaled = img_reshaped / 255
    # st.write(img_scaled)
    # return img_scaled



def remove_hair(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
    dst = dst[:, :, ::-1]
    # cv2.imshow('no hair image?', dst)
    return dst


def prepare_one_image_no_tl(img):
    tf_image = np.array(img)
    img_resized = np.resize(tf_image, (224, 224, 3))
    # st.write(img_resized.shape)
    img_resized = img_resized[:, :, ::-1]  # RGB to BGR
    img_reshaped = img_resized.reshape(
        (1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
    # st.write(img_reshaped.shape)
    img_scaled = img_reshaped / 255
    # st.write(img_scaled)
    return img_scaled
