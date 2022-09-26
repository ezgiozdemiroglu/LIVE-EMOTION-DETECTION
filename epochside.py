
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json

tf.__version__

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range= 0.2,
        zoom_range=0.2,
        horizontal_flip= True
)
training_set2 = train_datagen.flow_from_directory(
        'dataset/training_set2', 
        target_size=(64,64),
        batch_size=32,
        class_mode= 'categorical'
)

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set2= test_datagen.flow_from_directory(
    'dataset/test_set2', target_size=(64,64),
    batch_size= 32,
    class_mode= 'categorical'
)


# Building the CNN
cnn= tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu', input_shape=(64,64,3) ))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu' ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Adding third convolutional layer 
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu' ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#adding forth
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu' ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))



#Training the CNN
# Compiling the CNN
cnn.compile(optimizer= 'adam',  loss = 'categorical_crossentropy', metrics=['accuracy'])
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set2, validation_data=test_set2, epochs=10)



# serialize model to JSON
cnn_json = cnn.to_json()
with open("cnn.json", "w") as json_file:
    json_file.write(cnn_json)
# serialize weights to HDF5
cnn.save_weights("cnn.h5")
print("Saved cnn to disk")

