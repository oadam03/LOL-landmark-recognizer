import argparse
import PySimpleGUI as sg
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from io import BytesIO


# Path to your CSV file
csv_path = "balanced_train_v2.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

hierarchical_classes = sorted(df["hierarchical_label"].unique())
natural_classes = sorted(df["natural_or_human_made"].unique())

# Define the MultiTaskModel class
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_hierarchical_labels, num_natural_labels):
        super(MultiTaskModel, self).__init__()
        self.base = base_model
        in_features = self.base._conv_head.out_channels  # Match the feature size
        self.base._fc = nn.Identity()  # Replace classification head with identity
        self.head_hierarchical = nn.Linear(in_features, num_hierarchical_labels)
        self.head_natural = nn.Linear(in_features, num_natural_labels)

    def forward(self, x):
        features = self.base.extract_features(x)
        features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        hierarchical_out = self.head_hierarchical(features)
        natural_out = self.head_natural(features)
        return hierarchical_out, natural_out

# Load your trained model
MODEL_PATH = "LOLV3MODEL backup.pth"
device = torch.device("cuda")  # Use CPU; adjust to "cuda" if running on GPU
multi_task_model = torch.load(MODEL_PATH, map_location=device)
multi_task_model.eval()

# Define preprocessing transformations (should match your training pipeline)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Run predictions on the input image."""
    try:
        # Load image and preprocess
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Make predictions
        with torch.no_grad():
            hierarchical_output, natural_output = multi_task_model(image)
            hierarchical_pred = torch.argmax(hierarchical_output, dim=1).item()
            natural_pred = torch.argmax(natural_output, dim=1).item()

        return hierarchical_classes[hierarchical_pred], natural_classes[natural_pred]
    except Exception as e:
        return f"Error during prediction: {str(e)}", None


def display_image_with_prediction(image_path, prediction):
    """Display the image and prediction results in a PySimpleGUI window."""
    image = Image.open(image_path)
    image_data = np.array(image)
    height, width = image_data.shape[:2]
    aspect_ratio = width / height
    new_width, new_height = 640, int(640 / aspect_ratio)

    resized_image = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    sg_image = Image.fromarray(resized_image)

    # Convert resized image to displayable data
    with BytesIO() as output:
        sg_image.save(output, format="PNG")
        sg_image_data = output.getvalue()

    # Define layout
    layout = [
        [sg.Image(data=sg_image_data, key="-IMAGE-")],
        [sg.Text(f"Hierarchical Label: {prediction[0]}", font=('Helvetica, 32'))],
        [sg.Text(f"Natural Label: {prediction[1]}", font=('Helvetica, 32'))],
        [sg.Button("Exit")]
    ]

    window = sg.Window("Image Prediction", layout, finalize=True)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
    window.close()


def main():
    # GUI Layout
    layout = [
        [sg.Text("Upload an Image for Prediction")],
        [sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("Image Files", "*.png;*.jpg;*.jpeg"),))],
        [sg.Button("Predict"), sg.Button("Exit")]
    ]

    window = sg.Window("Image Classifier", layout, finalize=True)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
        if event == "Predict":
            image_path = values["-FILE-"]
            if not image_path:
                sg.popup("Please select an image file.", title="Error")
                continue

            prediction = predict_image(image_path)
            if prediction[1] is None:
                sg.popup(prediction[0], title="Error")
            else:
                display_image_with_prediction(image_path, prediction)

    window.close()


if __name__ == "__main__":
    main()
