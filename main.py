import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
import os
import random


label_predict = {
    'Apple___Apple_scab' : 'Ghẻ táo',
    'Apple___Black_rot' : 'Táo thối đen',
    'Apple___Cedar_apple_rust' : 'Rỉ sét táo tuyết tùng',
    'Apple___healthy' : 'Táo khỏe mạnh',
    'Cherry_(including_sour)___Powdery_mildew' : 'Anh đào phấn trắng',
    'Cherry_(including_sour)___healthy' : 'Anh đào khỏe mạnh',
    'Grape___Black_rot' : 'Nho sưng đen',
    'Grape___Esca_(Black_Measles)' : 'Nho sởi đen',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)' : 'Nho mục đốm lá',
    'Grape___healthy' : 'Nho khỏe mạnh',
    'Orange___Haunglongbing_(Citrus_greening)' : 'Cam vàng lá gân xanh',
    'Peach___Bacterial_spot' : 'Đào đốm vi khuẩn',
    'Peach___healthy' : 'Đào khỏe mạnh',
    'Strawberry___Leaf_scorch' : 'Dâu tây cháy lá',
    'Strawberry___healthy' : 'Dâu tây khỏe mạnh',
    'Tomato___Bacterial_spot' : 'Cà chua đốm vi khuẩn',
    'Tomato___Early_blight' : 'Cà chua mốc sương sớm',
    'Tomato___Late_blight' : 'Cà chua mốc sương',
    'Tomato___Leaf_Mold' : 'Cà chua nấm mốc lá',
    'Tomato___Septoria_leaf_spot' : 'Cà chua bệnh đốm lá Septoria',
    'Tomato___Spider_mites Two-spotted_spider_mite' : 'Cà chua côn trùng bám lá (Mối hai chấm)',
    'Tomato___Target_Spot' : 'Cà chua đốm mục tiêu',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus' : 'Cà chua Virus gây bệnh vàng lá',
    'Tomato___Tomato_mosaic_virus' : 'Cà chua Mosaic virus',
    'Tomato___healthy' : 'Cà chua khỏe mạnh'
 }
device = 'cpu'
# print(device)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
def predict_image(image_path, model, device):
    img = Image.open(image_path)
    img = transforms.Resize((256, 256))(img)
    xb = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        yb = model(xb)
    probs = torch.nn.functional.softmax(yb[0], dim=0)
    percent, preds = torch.max(probs, dim=0)
    predicted_label = list(label_predict.keys())[preds.item()]
    percent = percent * 100
    return predicted_label, round(percent.item(),2)
    

num_classes = len(label_predict)

model1 = torchvision.models.resnet50()
model1.fc = nn.Linear(model1.fc.in_features, num_classes)
model1.to(device)
model1.load_state_dict(torch.load('models\model_resnet50.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = torchvision.models.vgg16()
model2.classifier[6] = nn.Linear(model2.classifier[6].in_features, num_classes)
model2.to(device)
model2.load_state_dict(torch.load('models\model_vgg16.pth', map_location=torch.device('cpu')))
model2.eval()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('./index.html', percent=0.0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('./index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('./index.html', message='No selected file')

    if file:
        filename = file.filename
        file_path = f'static/{filename}'
        file.save(file_path)
        predicted_class1, percent1 = predict_image(file_path, model1, device)
        predicted_class2, percent2 = predict_image(file_path, model2, device)
        print(f'Resnet prediction: {predicted_class1} {percent1}')
        print(f'Vgg prediction: {predicted_class2} {percent2}')
        
        if percent1 >= percent2:
            predicted_class, percent = predicted_class1, percent1
        else:
            predicted_class, percent = predicted_class2, percent2

        # Lấy đường dẫn thư mục tương ứng với predicted_class
        class_directory = os.path.join('static', predicted_class)

        # Nếu thư mục tồn tại, thì lấy danh sách tất cả các tệp ảnh trong thư mục đó
        if os.path.exists(class_directory):
            images_in_class = [os.path.join(class_directory, img) for img in os.listdir(class_directory) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Nếu có ít nhất 3 ảnh, chọn ngẫu nhiên 3 ảnh
            if len(images_in_class) >= 3:
                similar_images = random.sample(images_in_class, 3)
                return render_template('./index.html',
                                       message=label_predict[predicted_class],
                                       percent = percent, selected_image_path=file_path,
                                       similar_images=similar_images, 
                                       predicted_class1=label_predict[predicted_class1],
                                       predicted_class2=label_predict[predicted_class2],
                                       percent1=percent1,
                                       percent2=percent2,
                                       )

    return render_template('./index.html', message='Failed to predict or find similar images', selected_image_path=file_path)


if __name__ == '__main__':
    app.run(debug=True)