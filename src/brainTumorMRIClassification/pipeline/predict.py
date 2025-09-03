import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("artifacts", "training", "brain_model.h5"))
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # Categorical types for the training data:
        # {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
        prediction = ""
        if result[0] == 0:
            prediction = "Glioma Detected"
        elif result[0] == 1:
            prediction = "Meningioma Detected"
        elif result[0] == 2:
            prediction = "No Tumor Detected"
        elif result[0] == 3:
            prediction = "Pituitary Tumor Detected"
        else:
            prediction = "Unknown"

        print(prediction)
        return [{"image": prediction}]
