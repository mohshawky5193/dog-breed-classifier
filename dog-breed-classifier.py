import face_recognition
import cv2
import argparse
import torch
from torch import nn
from torchvision import models,transforms
from PIL import Image

def face_detector(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(image,number_of_times_to_upsample=0,model='cnn')
    return len(faces)>0

def resnet50_predict(img_path):
    resnet50 = models.resnet50(pretrained=True)
    image = Image.open(img_path)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean,std)])
    image_tensor = image_transforms(image)
    image_tensor.unsqueeze_(0)
    resnet50.eval()
    output = resnet50(image_tensor)
    _,classes = torch.max(output,dim=1)
    return classes.item()

def resnet50_dog_detector(image_path):
    class_idx = resnet50_predict(image_path)
    return class_idx >= 151 and class_idx <=268

def get_dog_breed_classifier():
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    model.classifier=nn.Sequential(nn.Linear(1024,512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                       nn.Linear(512,133))
    return model


def predict_breed_transfer(img_path,model):
    # load the image and return the predicted breed
    f = open('dog_breeds.txt')
    class_names = f.readlines()
    class_names = [class_name[:-1] for class_name in class_names]
    image=Image.open(img_path)
    mean_train_set,std_train_set = [0.487,0.467,0.397],[0.235,0.23,0.23]
    image_transforms= transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_train_set,std_train_set)])
    image_tensor = image_transforms(image)
    image_tensor.unsqueeze_(0)
    model.eval()
    output = model(image_tensor)
    _,class_idx=torch.max(output,dim=1)
    return class_names[class_idx]
def check_image_file(path):
    image_extensions=['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']
    if path.split('.')[1].lower() in image_extensions:
        return True
    else:
        return False
def run_app(img_path,out_path):
    """
    if the image passed is of human places a dog filter on face detected
    if the image passed is of dog writes the prediction of dog breed
    if neither print an error message
    """
        
    if(resnet50_dog_detector(img_path)):
        model_dog_classifier = get_dog_breed_classifier()
        model_dog_classifier.load_state_dict(torch.load('model_transfer.pt',map_location='cpu'))
        dog_breed=predict_breed_transfer(img_path,model_dog_classifier)
        dog_image = cv2.imread(args.input_path)
        width,height =dog_image.shape[:2]
        cv2.putText(dog_image, dog_breed, (int(0*width), int(0.2*height)), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA) 
        cv2.imwrite(out_path,dog_image)
        print('This a picture of dog of breed:{}'.format(dog_breed))
    elif(face_detector(img_path)):
        image = cv2.imread(args.input_path)
        dog_face = cv2.imread('dog-face.png',-1)
        image_dog_face=dog_face[:,:,0:3]
        face_locations=face_recognition.face_locations(image,model='cnn')
        location=face_locations[0]
        mask =dog_face[:,:,3]
        mask_inverse = cv2.bitwise_not(mask)
        roi =image[location[3]:location[1],location[0]:location[2],:]
    
        length_width=roi.shape[:2]
        length_width = tuple(length_width)
        mask_inverse_resized = cv2.resize(mask_inverse,length_width,interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask,length_width,interpolation=cv2.INTER_AREA)
        image_dog_face_resized = cv2.resize(image_dog_face,length_width,interpolation=cv2.INTER_AREA)
        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inverse_resized)
        roi_fg = cv2.bitwise_and(image_dog_face_resized,image_dog_face_resized,mask=mask_resized)
        image[location[3]:location[1],location[0]:location[2]] = cv2.add(roi_bg,roi_fg)
        cv2.imwrite(out_path,image)
    else:
        print ('This image does not contain neither human face nor dog')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',help='path to the input image file')
    parser.add_argument('--output_path',help='path to the output image file',default='output.jpg')
    args = parser.parse_args()
    if (check_image_file(args.input_path)):
        run_app(args.input_path,args.output_path)
    else:
        print('This is not a valid image file please use another file')

