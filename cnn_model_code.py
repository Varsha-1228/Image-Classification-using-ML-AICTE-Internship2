# CNN model implementation using Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialize the CNN model
cnn = Sequential()

# Add convolutional and pooling layers
cnn.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32,32,3)))
cnn.add(MaxPooling2D((2,2)))

cnn.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
cnn.add(MaxPooling2D((2,2)))

cnn.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
cnn.add(MaxPooling2D((2,2)))

cnn.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
cnn.add(MaxPooling2D((2,2)))

# Flatten and add dense layers
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))  # Hidden layer
cnn.add(Dropout(0.3))
cnn.add(Dense(10, activation='softmax'))  # Output layer
