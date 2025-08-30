import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from model import build_model

# Optionally, load weights if you have a trained model
# model = build_model()
# model.load_weights('path_to_weights.h5')

model = build_model()


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_and_visualize(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img, verbose=0)
    classes = ['Anemic', 'Non-Anemic']
    predicted_class = classes[int(prediction.round())]
    plt.imshow(image.load_img(img_path))
    plt.title(f'Predicted: {predicted_class}', fontsize=10)
    plt.axis('off')
    plt.show()


# Example usage:
input_path = "/kaggle/input/clean-augmented-anemia-dataset/New_Augmented_Anemia_Dataset/Conjuctiva/Testing/Anemic/Anemic-001FV_aug13.png"
predict_and_visualize(input_path)
