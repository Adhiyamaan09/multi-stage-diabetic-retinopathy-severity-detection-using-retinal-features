import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from python_scripts.const import MODEL_PATH, NUM_OF_CLASSES, IMG_SIZE, DR_CLASSES, COLOR_MAP
from python_scripts.model import ConvNetModel
from python_scripts.utils import convert_plt_figure_to_base64


matplotlib.use('agg')

class ImagePrediction():
    device = "cpu"
    model = None
    transforms = None 

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNetModel().to(self.device)
    
    # ✅ Load model on the correct device
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))

        self.transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
    ])

    
    def getPrediction(self, image):
        image = self.transforms(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(image)
            outputs = torch.softmax(outputs, dim = 1)
            probabilities = [t.item() * 100 for t in outputs.squeeze()]

        dr_class_idx = np.argmax(probabilities)
        dr_class_probability = probabilities[dr_class_idx]
        dr_class = DR_CLASSES[dr_class_idx]
        
        plt.title("Diabetic Retinopathy")
        plt.ylim(0, 100)
        plt.bar(x = DR_CLASSES, height = probabilities, color = COLOR_MAP[dr_class])
        fig = plt.gcf()
        img_str = convert_plt_figure_to_base64(fig)
        plt.close()

        return dr_class, dr_class_probability, img_str