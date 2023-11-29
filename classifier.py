from taipy.gui import Gui
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np

classes = {
   0 : "🐵 Mantled_Howler",
   1 : "🐒 Patas_Monkey",
   2 : "🙈 Bald_Uakari",
   3 : "🙉 Japanese_Macaque",
   4 : "🙊 Pygmy_Marmoset",
   5 : "🐵 White_Headed_Capuchin",
   6 : "🐒 Silvery_Marmoset",
   7 : "🙈 Common_Squirrel_Monkey",
   8 : "🙉 Black_Headed_Night_Monkey",
   9 : "🙊 Nilgiri_Langur"
}

# Load the model
try:
    model = torch.load('best_model.pth')
    print("🎉 Model loaded successfully!")
except Exception as e:
    print(f"😞 Error loading model: {e}")
    model = None

image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify(model, image_transforms, image_path, classes):
    model.eval()

    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    # Apply softmax to output to get probabilities
    probabilities = F.softmax(output.data, dim=1)

    # Get the maximum probability
    max_probability = torch.max(probabilities).item()

    # Get the predicted class
    top_pred = classes[predicted.item()]
    top_prob = max_probability

    return top_pred, top_prob

content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|

<|{"logo.png"}|image|width=50vw|>

<|{content}|file_selector|extensions=.jpg|>
🖼️ Please select an image to upload 

<|{pred}|>


<|{img_path}|image|>




<|{prob}|indicator|value={prob}|min=0|max=100|width=15vw|>
"""


def on_change(state, var_name, var_val, class_names):
    if var_name == "content":
        top_pred, top_prob = classify(model, image_transforms, var_val, classes)
        state.prob = round(top_prob * 100)
    
        state.img_path = var_val
        state.pred = "🔍 This is a " + top_pred

app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)
