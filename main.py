from fastapi import FastAPI, Body, Request, File, UploadFile, Form, BackgroundTasks, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms ,models
import torch.optim as optim

import datetime
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI(debug=True)
templates = Jinja2Templates(directory= "templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cpu")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Prepare class Model
class ModeL(nn.Module):
    def __init__(self, n_classes,device= 'cuda'):

        super(ModeL, self).__init__()
        self.model = self._creat_Model(n_classes).to(device)  

    def _creat_Model(self,out_features, pretrained = False):

        model= models.efficientnet_b2(pretrained=pretrained)
        model.classifier =  nn.Sequential(
            nn.Linear(in_features=1408, out_features=out_features, bias=True),
            nn.LogSoftmax(dim = 1))

        return model

    def forward(self, x):
        x = self.model(x)
        return x

# Your transformation
transform= transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5694, 0.4460, 0.3912), (0.2734, 0.2435, 0.2370))
                                                            ])

# create and load the model
model =ModeL(4).to(device) 

state = torch.load("AFF_4-Class_Acc equal 90.6 train and 76.8 val .pt",map_location=device)
model.load_state_dict(state['state_dict'],strict=True)
model.eval()

# Function to predict the emotion
def img_ER (img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    inputs = transform(img)

    final_image = torch.unsqueeze(inputs, 0)
    dataa=final_image.type(torch.FloatTensor).to(device)
    outputs = model(dataa)
    pred = F.softmax(outputs,dim=1)
    prediction = torch.argmax(pred)

    if ((prediction) == 0):
        status = "Angry"
    elif ((prediction) == 1):
        status = "Happy"  
    elif ((prediction) == 2):
        status = "Sad"   
    elif ((prediction) == 3): 
        status = "Surprise"

    return (status)

# Flask function
def gen_frames():  
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # cv2.CAP_DSHOW
    last_time = datetime.datetime.now()
    frames = 0
    with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
        
        while True:
            success, image = camera.read()  # read the camera frame
            if not success:
                print("Ignoring empty camera frame.")
                break

            else:
                frames += 1
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                image_h = image.shape[0]
                image_w = image.shape[1]


                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for face_no, detection in enumerate(results.detections):
                        try:
                            face_bounding = detection.location_data.relative_bounding_box
                            h = int((face_bounding.height) * image_h)
                            xmin = int(((face_bounding.xmin) * image_w)+10)
                            w = int((face_bounding.width) * image_w)
                            ymin = int(((face_bounding.ymin) *image_h)+10)

                            face_for_emo = image[ymin:ymin+h , xmin:xmin+w]

                            emo = img_ER(face_for_emo)

                            cv2.putText(image,emo,(xmin+int(w/10),ymin+int(h/10)) ,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255),2)

                            mp_drawing.draw_detection(image, detection)
                        except:
                            pass
                    
            delta_time = datetime.datetime.now() - last_time
            elapsed_time = delta_time.total_seconds()
            cur_fps = np.around(frames / elapsed_time, 1)

            cv2.putText(image, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {'request':request})

@app.get('/video_feed')
async def video_feed():
    # return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port =8000, reload=True)