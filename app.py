import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import gradio as gr
from models import COVIDCNN

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load Model ----
def load_model():
    model = COVIDCNN(num_classes=2)
    model.load_state_dict(torch.load('models/best_covid_model.pth', map_location=device))
    model.to(device)   # move model to GPU if available
    model.eval()
    return model

model = load_model()

# ---- Transform ----
class ConvertToGrayscale:
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        if image.mode != 'L':
            image = image.convert('L')
        return image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    ConvertToGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5170, std=0.2072)
])

# ---- Prediction Function ----
def predict(img):
    x = transform(img).unsqueeze(0).to(device)  # send input to GPU
    with torch.no_grad():
        preds = model(x)
        classes = ["Normal", "COVID-19"]
        probs = preds.softmax(1)[0].cpu()  # bring back to CPU for display
        return {c: float(probs[i]) for i, c in enumerate(classes)}

# ---- Gradio Interface ----
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=gr.Label(num_top_classes=2),
    title="🫁 COVID-19 X-ray Classifier",
    description="Upload an image and get instant prediction."
).launch()
