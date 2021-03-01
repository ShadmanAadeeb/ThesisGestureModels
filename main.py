import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
#import tensorflow as tf
#from tensorflow.keras.models import Model, load_model
class ModelController():
        
    def __init__(self,tkInterWindow):
        #Control Variables
        self.controllerFlag=0
        self.interval=20
        # controller flag =0 means no model is running
        # controller flag =1 means hand gesture model is running
        # controller flag =2 means nose face model is running                             
                       
        #Setting up GUI and webcam
        self.tkInterWindow=tkInterWindow                
        self.camera=cv2.VideoCapture(1)                
        self.width=self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height=self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        #Face nose variables
        # 0 means face
        # 1 means nose
        self.faceOrNose=0
        self.faceCenterCoOrdinate=None 
        self.noseCoOrdinate=None  

        #Setting up the canvases
        self.setUpTheCanvases()
        #Inserting all the buttons inside the tkinterWindow
        self.setUpTheButtons()
        #Loading all the models
        self.loadAllModels()

    def loadAllModels(self):
        self.noseTeethModel=load_model('./Weights/teeth_close_open.h5')
        self.noseTeethModel.summary()

    def setUpTheCanvases(self):        
        #Setting up the main canvase
        self.mainCanvas=tk.Canvas(self.tkInterWindow,width=self.width,height=self.height)
        self.mainCanvas.pack(side=tk.LEFT, padx=5, pady=5)        
        self.blackImgArray=np.ones( (int(self.height),int(self.width))  )*0
        self.image = Image.fromarray(self.blackImgArray) # to PIL format
        self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format                                
        self.mainCanvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        #Setting up the main canvas ends
    
    def setUpTheButtons(self):
        #Model running buttons
        self.buttonsFrame=tk.Frame(master=self.tkInterWindow,relief=tk.RAISED,borderwidth=1)        
        #Making the Quit Button
        self.quitButton = tk.Button(self.buttonsFrame, text= "Quit", command= self.tkInterWindow.destroy)
        self.quitButton.pack(side=tk.BOTTOM, padx=5, pady=5)
        #Stop Button
        self.stopButton=tk.Button(self.buttonsFrame,text="Stop Button",command=None)
        self.stopButton.pack(side=tk.BOTTOM,padx=5,pady=5)
        #Making the Hand Gesture Runner Button
        self.handGestureButton=tk.Button(self.buttonsFrame,text="Hand Gesture",command=None)
        self.handGestureButton.pack(side=tk.BOTTOM,padx=5,pady=5)
        #Making the Nose Teeth Runner Button
        self.NoseTeethButton=tk.Button(self.buttonsFrame,text="Nose Teeth",command=self.preRunFaceNoseModel)
        self.NoseTeethButton.pack(side=tk.BOTTOM,padx=5,pady=5)

        #Packing the button frame
        self.buttonsFrame.pack(side=tk.RIGHT,padx=5,pady=5)
    
    def preRunFaceNoseModel(self):
        #Disabling the other buttons
        self.NoseTeethButton["state"]="disable"
        self.handGestureButton["state"]="disable"
        #Flag indicating  facenose model                        
        self.controllerFlag=2
        #Running face nose model
        self.runFaceNoseModel()
    
    def  runFaceNoseModel(self):
        #Reading img from camera                
        ret, frame = self.camera.read()                
        #Running Yolo Algorithm it will generate the nose detected image named self.image        
        self.runYoloAlgorithmFace(frame)        
        self.mainCanvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        if self.controllerFlag==2:
            self.tkInterWindow.after(self.interval, self.runFaceNoseModel)
        else:
            self.faceOrNose=0
            return
    
    def runYoloAlgorithmFace(self,img):                
        # Processing The Image and making it into a blob
        img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392 , (288, 288), (0, 0, 0), True, crop=False)
        #making the yolo nn architecture
        if self.faceOrNose==0:
            net = cv2.dnn.readNet("Weights\yolov4-tiny-face_best.weights", "cfg_file\yolov4-tiny-testing.cfg")            
        elif self.faceOrNose==1:
            net = cv2.dnn.readNet("Weights\yolov4-tiny-nose_best.weights", "cfg_file\yolov4-tiny-testing.cfg")        
        #Getting reference to the output layers
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #Making an array to contain the name of objects in final layer
        classes = [""]                                
        #Passing the blob to the neural network
        net.setInput(blob)
        #Collecting the feature maps from the output layers
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []        
        for out in outs:            
            for detection in out:
                #The three lines below is used for finding confidence value
                scores=detection[5:] 
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence>0.3:
                    #It means we are considering a detected object                    
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)                                            
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)                                                
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)            
            # The boxes array contain the detected boxes            
            # We further get the indexes of boxes after Non Max Suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)                        
            font = cv2.FONT_HERSHEY_PLAIN            
            #Now drawing the boxes from the images            
            
            for i in range(len(boxes)):
                if i in indexes:
                    try:
                        x, y, w, h = boxes[i]
                        #print("x1,y1 is= "+str(x)+", "+str(y))
                        label = str(classes[class_ids[i]])
                        #print("Label ",label)
                        color = (0,0,0)                                                
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
                        
                        croppedImg = img[y:y+h, x:x+w]
                        croppedImg=cv2.resize(croppedImg,(100,100))
                        teethx1=int(x+0.15*w)
                        teethy1=int(y+h*0.60)
                        teethx2=int(x+0.85*w)
                        teethy2=int(y+0.78*h)
                        #cv2.rectangle(img, (teethx1, teethy1), (teethx2, teethy2), (255,0,0), 1)                            
                        croppedImg=img[teethy1:teethy2,teethx1:teethx2]
                        croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
                        croppedImg=cv2.resize(croppedImg,(224,224))
                        
                        #cv2.imshow("croppedImg",croppedImg)
                        #If this is for face then we store face co-ordinate
                        if self.faceOrNose==0:
                            self.faceCenterCoOrdinate=(  int(x+w/2)  ,  int(y+h/2) )
                            croppedImg = np.stack((croppedImg,)*3, axis=-1)        
                            croppedImg=np.reshape(croppedImg,(1,224,224,3))
                            croppedImg=croppedImg/255
                            y=self.noseTeethModel.predict(croppedImg)                                   
                                
                        #else if this is for nose we store the nose co=ordinate
                        elif self.faceOrNose==1:
                            self.noseCoOrdinate=(  int(x+w/2)  ,  int(y+h/2) )
                        #cv2.imshow("CroppedImage",croppedImg)

                        #Now Performing the cursor displacement function
                        self.cursorMovementWithNose()
                    except:
                        pass
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # to RGB
            self.image = Image.fromarray(self.image) # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format  
        #Next time choose the other
        if self.faceOrNose==0:
            self.faceOrNose=1
        elif self.faceOrNose==1:
            self.faceOrNose=0       


    def cursorMovementWithNose(self):
        
        if self.noseCoOrdinate is not None and self.faceCenterCoOrdinate is not None:
            xDisplacement=self.faceCenterCoOrdinate[0]-self.noseCoOrdinate[0]
            yDisplacement=self.faceCenterCoOrdinate[1]-self.noseCoOrdinate[1]
            # print("*******************************************")
            # print(f"face tuple is {self.faceCenterCoOrdinate}")
            # print(f"nose tuple is {self.noseCoOrdinate}")
            # print(f"xdisplacement is {xDisplacement}")
            # print(f"ydisplacement is {yDisplacement}")
            # print("*******************************************")

    



if __name__ == "__main__":
    mainWindow=tk.Tk()
    modelController=ModelController(mainWindow)
    mainWindow.mainloop()

