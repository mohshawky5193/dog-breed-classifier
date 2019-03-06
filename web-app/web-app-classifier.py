from flask import Flask,request,jsonify,render_template
import os
import torch
from torchvision import models,transforms
import numpy as np
from PIL import Image
import io
import face_recognition
import cv2
import face_recognition
from torch import nn
def face_detector(img):
    
    faces = face_recognition.face_locations(img)
    return len(faces)>0
use_cuda = torch.cuda.is_available()
def resnet50_predict(pil_image):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    # Image Resize to 256
    resnet50 = models.resnet50(pretrained=True)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean,std)])
    image_tensor = image_transforms(pil_image)
    image_tensor.unsqueeze_(0)
    resnet50.eval()
    if use_cuda:
        image_tensor=image_tensor.cuda()
    output = resnet50(image_tensor)
    _,classes= torch.max(output,dim=1)
    return classes.item() # predicted class index

def dog_detector(pil_image):
    ## TODO: Complete the function.
    class_dog=resnet50_predict(pil_image)
    return class_dog >= 151 and class_dog <=268 # true/false

def predict_breed_transfer(pil_image,model):
    # load the image and return the predicted breed
    mean_train_set,std_train_set = [0.487,0.467,0.397],[0.235,0.23,0.23]
    
    image_transforms= transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_train_set,std_train_set)])
    image_tensor = image_transforms(pil_image);
    image_tensor.unsqueeze_(0)
    if use_cuda:
        image_tensor = image_tensor.cuda()
    model.eval()
    output = model(image_tensor)
    _,class_idx=torch.max(output,dim=1)
    f = open('dog_breeds.txt')
    class_names = f.readlines()
    class_names = [class_name[:-1] for class_name in class_names]
    return class_names[class_idx]

app = Flask(__name__, static_url_path='/static')
@app.route('/about')
def render_about_page():
    return render_template('about.html')
@app.route('/')
def render_page():
    return render_template('dog-web-app.html')

@app.route('/uploadajax',methods=['POST'])
def upload_file():
    """
    retrieve the image uploaded and make sure it is an image file
    """
    file = request.files['file']
    image_extensions=['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']
    if file.filename.split('.')[1] not in image_extensions:
        return jsonify('Please upload an appropriate image file')
    """
    Load the trained densenet model
    """
    model_transfer = models.densenet121(pretrained=False)
    model_transfer.classifier=nn.Sequential(nn.Linear(1024,512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                       nn.Linear(512,133))
    model_transfer.load_state_dict(torch.load('model_transfer.pt',map_location='cpu'))
    """
    Load variables needed to detect human face/dog pil image for dog detection and breed prediction and
    numpy for face detection
    """
    image_bytes = file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if (dog_detector(pil_image)):
        dog_breed = predict_breed_transfer(pil_image,model_transfer)
        return jsonify ('This a dog picture of breed:{}'.format(dog_breed))
    elif (face_detector(img_np)):
        dog_breed = predict_breed_transfer(pil_image,model_transfer)
        return jsonify('Hello Human You resemble dog breed of {}'.format(dog_breed))
    else:
        return jsonify('This pic doesn\'t have a human face or a dog')
if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))