MODEL_PATH = "./model/ConvNet_model.pt"
IMG_SIZE = 224
COLOR_MAP = {
    "No DR": "green",
    "Mild": "lightgreen",
    "Moderate": "indianred",
    "Severe": "orangered",
    "Proliferative": "red"
}

DR_CLASSES = list(COLOR_MAP.keys())
NUM_OF_CLASSES = len(DR_CLASSES)