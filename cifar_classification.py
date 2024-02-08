from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the input values
x_train = x_train/255.0
x_test = x_test/255.0

#print(x_train)

# One-hot encode categories
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()

# Convolutional Layer
model.add(Conv2D(32, (3,3), padding ='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))

# Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Another Convolutional Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))

# Another Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten Layer
model.add(Flatten())

# Dense Layer
model.add(Dense(64))
model.add(Activation('relu'))


# Output Layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
hiatory = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    shuffle=True)
         

model.save('cifar_CNN.h5')
         
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

