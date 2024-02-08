from PIL import Image
from tensorflow import keras
import numpy as np
from keras.models import load_model



image_path = 'data/dog4.jpeg'
image = Image.open(image_path)

# Resize the image
image = image.resize((32,32))

# Convert the image to array
image_array = np.array(image)

# Normalize the input
image_array = image_array/255.0

# If your model expects a batch of images, you need to add an extra dimension
image_array = np.expand_dims(image_array, axis=0)

# Load model
model = load_model('cifar_CNN.h5')

# Make prediction using model
predictions = model.predict(image_array)

# Extract index of label with high dist. value
predicted_class = np.argmax(predictions[0])

# Create list of labels
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Map index of highest prob. to appropriate label
predicted_class_name = class_names[predicted_class]

print("Predicted Class: ", predicted_class_name)

