import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template

label_predict = [
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Raspberry___healthy',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

def predict_image(image_path, model, device):
    img = Image.open(image_path)
    xb = transforms.ToTensor()(img).unsqueeze(0).to(device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    print(yb)
    return label_predict[preds[0].item()]
    
device = 'cpu'
# print(device)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = torchvision.models.resnet50(pretrained=True)
num_classes = len(label_predict)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

model.load_state_dict(torch.load('D:\DHCNHN\HK7\ĐỒ ÁN CHUYÊN NGÀNH\BTL\CT\DoAnChuyenNganh\model.pth', map_location=torch.device('cpu')))
model.eval()

# img = Image.open('TomatoEarlyBlight4.jpg')

# print( 'Predicted:', predict_image('C:\Project\DoAnChuyenNganh\PotatoHealthy1.JPG', model, device))
# dummy_input = torch.zeros(1, 3, 256, 256)
# torch.onnx.export(model, dummy_input, 'onnx_model.onnx', verbose=True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('./index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('./index.html', message='No selected file')

    if file:
        filename = file.filename
        file.save(filename)
        predicted_class = predict_image(filename, model, device)
        return render_template('./index.html', message='Predicted Class: {}'.format(predicted_class))

if __name__ == '__main__':
    app.run(debug=True)