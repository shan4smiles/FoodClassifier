import tensorflow as tf
import pathlib
import pickle

# Load the model from a Pickle file
import tensorflow as tf

# Load the Keras model
loaded_model = tf.keras.models.load_model('my_model_05.keras')


def knowAccuracy():
    # Load your evaluation dataset (similar to how you loaded your training dataset)
    img_height, img_width = 180, 180
    batch_size = 32
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        r'C:\Users\DELL\PycharmProjects\ResNet50\dataset\validation',
        seed=123,
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Evaluate the loaded model on the evaluation dataset
    evaluation = loaded_model.evaluate(val_ds)

    # The 'evaluation' variable will contain the loss and accuracy of the model
    loss, accuracy = evaluation

    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")


# Load class names from a text file
with open('folder_names.txt', 'r') as file:
    class_names = file.read().splitlines()

# Load and preprocess an image for prediction
image_path = 'soup.jpg'  # Replace with the path to your image
img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch of 1 image

# Make predictions
predictions = loaded_model.predict(img_array)
predicted_class_index = tf.argmax(predictions, axis=1)
predicted_class_name = class_names[predicted_class_index[0]]

print("Predicted class name:", predicted_class_name)
