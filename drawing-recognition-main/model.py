#Import modules
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Define a custom callback to stop training once a certain accuracy is reached
class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #Check if current accuracy is greater than 0.97
        if logs.get("accuracy") > 0.97:
            #If accuracy is greater than 0.97, print message and stop training
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True
callback = Mycallback()
#Load the MNIST dataset from keras.datasets
data = tf.keras.datasets.mnist

#Split the dataset into training and testing sets
(training_images, training_labels), (test_images, test_labels) = data.load_data()

#Define an ImageDataGenerator for training data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1/255,
    width_shift_range=0.3,
    height_shift_range=0.3
)

#Reshape the training images to have a fourth dimension (for compatibility with the CNN) and create a generator
training_images = training_images.reshape(60000, 28, 28, 1)
train_generator = train_datagen.flow((training_images,training_labels),batch_size=32)

#Define an ImageDataGenerator for validation data augmentation
validation_datagen = ImageDataGenerator(
    rescale=1/255,
    width_shift_range=0.3,
    height_shift_range=0.3)

#Reshape the testing images to have a fourth dimension (for compatibility with the CNN) and create a generator
test_images = test_images.reshape(10000, 28, 28, 1)
validation_generator = validation_datagen.flow(test_images, test_labels, batch_size=32)

#Define a Sequential model for the CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(132, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Train the model with data generators, and stop training if callback is triggered
model.fit(train_generator,validation_data=validation_generator,steps_per_epoch = len(training_images) //32 , epochs=100, callbacks=[callback])

#Evaluate the trained model on the testing dataset
model.evaluate(test_images,test_labels)

#Save the trained model to disk
model.save("final-cnn-digits-model")


