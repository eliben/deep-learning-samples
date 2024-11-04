import time
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Info on the format of CIFAR10 data:
# * https://keras.io/api/datasets/cifar10/
# * https://www.cs.toronto.edu/~kriz/cifar.html

model = models.load_model("trained-weights.keras")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print(test_images.shape, test_labels.shape)

label_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# for small predicts, calling the model object directly works
# Alternatively, can call predict:
#   model.predict(test_images[:1])
# Doc: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Predict the first 20 images; softmax converts them into probabilities,
# and we use argmax to find the most likely class for each one.
start = time.time()
num = 20
prediction = model(test_images[:num])
probs = tf.nn.softmax(prediction)
predindices = tf.argmax(probs, axis=1).numpy()
print(f"Prediction took {time.time() - start:.2f} seconds")

for i in range(num):
    predidx = predindices[i]
    testidx = test_labels[i][0]
    plt.imshow(test_images[i])
    plt.savefig(f"test_image_{i}.png")

    print(
        f"{i:2d} Predicted: {label_classes[predidx]}, Actual: {label_classes[testidx]}"
    )
