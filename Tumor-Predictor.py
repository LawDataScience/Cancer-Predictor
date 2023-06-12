import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import model_from_json
from pathlib import Path

cnn_path = Path("./cnn.h5")
cnn_path_json = Path("./cnn.json")

if not cnn_path:
    # processing the training dataset
    train_data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_data_generator.flow_from_directory('brain_tumor_dataset', target_size=(64, 64), batch_size=32,
                                                        class_mode='binary')
    # processing the test dataset
    test_data_generator = ImageDataGenerator(rescale=1./255)
    test_set = test_data_generator.flow_from_directory('test_dataset', target_size=(64, 64), batch_size=32,
                                                        class_mode='binary')
    # Initialize the CNN
    cnn = tf.keras.models.Sequential()
    # Adding a convolutional layer
    cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    # Add a pooling layer
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    # Adding another convolutional & pooling layer
    cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    # Adding a flattening layer
    cnn.add(tf.keras.layers.Flatten())
    # Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # Compile the Network
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training and Testing the Network
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)
    # save the model to disk for later
    cnn_to_json = cnn.to_json()
    with open("cnn.json", "w") as json_file:
        json_file.write(cnn_to_json)
    cnn.save_weights("cnn.h5")
    print("saved model to disk")

# processing the test image
test_image = image.load_img('Y52.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
# load in the model
json_file = open("cnn.json", "r")
loaded_cnn_json = json_file.read()
json_file.close()
loaded_cnn = model_from_json(loaded_cnn_json)
loaded_cnn.load_weights("cnn.h5")
print("loaded model from disk")
result = loaded_cnn.predict(test_image)
if result[0][0] == 1:
    prediction = "Unfortunately it's Cancer"
else:
    prediction = "it's not cancer"
print(prediction)
