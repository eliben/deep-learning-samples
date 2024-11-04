import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.load_model("trained-weights.keras")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print(test_images.shape, test_labels.shape)

# for small predicts, calling the model object directly works
# Alternatively, can call predict:
#   model.predict(test_images[:1])
# Doc: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Predict the first 20 images; softmax converts them into probabilities,
# and we use argmax to find the most likely class for each one.
num = 20
prediction = model(test_images[:num])
probs = tf.nn.softmax(prediction)
predindices = tf.argmax(probs, axis=1).numpy()

for i in range(num):
    print(f"Predicted: {predindices[i]}, Actual: {test_labels[i]}")
