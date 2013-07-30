import numpy.core.multiarray
import cv2
import time
import numpy as np
import workflow
import face_detection

capture = cv2.VideoCapture(-1)

def getFrame():
    _,frame = capture.read()
    frame=cv2.flip(frame,1)
    return frame

def imageRegistration(frame,histeq=False):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if histeq:
        gray = cv2.equalizeHist(gray)
    return gray

def main():
    import settings
    from cv2 import cv
    cv2.namedWindow("frame")
    cv.MoveWindow("frame",100,0)
    #cv2.namedWindow("gray")
    #cv.MoveWindow("gray",800,0)
    cv2.namedWindow("fps")
    cv.MoveWindow("fps",100,600)
    fd = face_detection.OpenCVFaceDetection(modelname = "../models/FD")
    modelag = workflow.Workflow('../models/age_gender_v0.1')
    modelsm = workflow.Workflow('../models/smile_v0.0')
    modelgl = workflow.Workflow('../models/glasses_v0.0')
    while(1):
        frame = getFrame()
        face_meta = []
        t0=time.time()
        faces=fd.run(frame)
        fps = 1.0/(time.time()-t0)
        for tmp_feature in modelag.features:
            del modelag.features[tmp_feature][1][:]
        del modelsm.features['smile'][1][:]
        del modelgl.features['glasses'][1][:]
        for face in faces:
            face_meta.append(modelag.execute(face.copy()))
            face_meta[-1].update(modelsm.execute(face))
            face_meta[-1].update(modelgl.execute(face))
        #cv2.imshow("gray",gray)
        fd.visualizeCurrentFaces(frame)
        cv2.imshow("frame",frame)
        results_window = np.zeros((400,300),dtype=np.uint8)+255
        i=0
        for face in face_meta:
            gender = settings.gender_map[face['gender'][0]]
            cv2.putText(results_window,"  Gender:    %s, conf = %2.2g"%(gender,face['gender'][1]),(10,settings.person_start_offset + i*settings.person_vertical_offset),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            age,uncer = face['age'][0],face['age'][1]
            cv2.putText(results_window,"  Age:       %2.2g, std = %2.2g"%(age,uncer),(10,settings.person_start_offset + i*settings.person_vertical_offset+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            smile = face['smile'][0]
            cv2.putText(results_window,"  Smile:     %2d%%"%(int(smile*100)),(10,settings.person_start_offset + i*settings.person_vertical_offset+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            glasses = settings.glasses_map[int(face['glasses'][0])]
            cv2.putText(results_window,"  Glasses:   %s"%(glasses),(10,settings.person_start_offset + i*settings.person_vertical_offset+60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            i+=1
        cv2.putText(results_window, "FPS:    %4.4g"%fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
        cv2.imshow("fps",results_window)
        key = cv2.waitKey(1);
        if key==27:
            break
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()

if __name__ == "__main__":
    main()
