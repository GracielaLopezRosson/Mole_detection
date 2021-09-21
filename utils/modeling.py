import pickle
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Dropout


def instantiate_model(X_train, X_val, y_train, y_val):
    model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    new_model = Sequential()
    new_model.add(model)
    new_model.add(BatchNormalization())
    new_model.add(Flatten())
    new_model.add(Dense(64, activation='relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(7, activation='softmax'))

    new_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = new_model.fit(X_train, y_train,
                            epochs=3,
                            batch_size=20,
                            validation_data=(X_val, y_val)
                            )

    new_model.save('model/model.h5')
    with open('model/history', 'wb') as filepath:
        pickle.dump(history.history, filepath)

    return history.history, new_model
