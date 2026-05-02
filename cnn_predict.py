import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.weights.h5", compile=False)

# Read image path
img_path = sys.argv[1]

img = cv2.imread(img_path)

if img is None:
    print("ERROR")
    sys.exit()

img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.reshape(img, (1, 64, 64, 3))

prediction = model.predict(img, verbose=0)[0][0]

if prediction > 0.5:
    print("OPEN")
else:
    print("CLOSED")