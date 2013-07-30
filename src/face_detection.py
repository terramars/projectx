import cv2
import workflow
import preprocessing
import os
import time

class OpenCVFaceDetection(object):
    preprocessing_flow = None
    face_cascade = None
    face_workflow = None
    use_static = False
    base_sub_region = None
    current_faces = []
    last_faces_1 = []
    last_faces_2 = []
    min_size = (30,30)
    scaling = 1.2
    neighbors = 2
    flags = cv2.CASCADE_DO_CANNY_PRUNING | cv2.CASCADE_SCALE_IMAGE
    
    def __init__(self,use_static = False, base_sub_region = [0,0,1,1], modelname = "../models/haarcascade_frontalface_alt.xml",  face_workflow_name = None, use_histeq = False, use_gammacorr = False, use_contrasteq = False):
        self.use_static = use_static
        if use_static:
            self.flags |= cv2.CASCADE_FIND_BIGGEST_OBJECT
        self.base_sub_region = base_sub_region
        self.preprocessing_flow = preprocessing.PreprocessingFlow( [preprocessing.Grayscale()] )
        if use_histeq:
            self.preprocessing_flow.append(preprocessing.HistogramEQ())
        if use_gammacorr:
            self.preprocessing_flow.append(preprocessing.GammaCorrection())
        if use_contrasteq:
            self.preprocessing_flow.append(preprocessing.ContrastEQ())
        if use_gammacorr or use_contrasteq:
            self.preprocessing_flow.append(preprocessing.ConvertToUInt8())
        self.face_cascade = cv2.CascadeClassifier(modelname)
        if face_workflow_name:
            self.face_workflow = workflow.Workflow(face_workflow_name)
    
    def setCurrentFaces(self,gray):
        self.last_faces_2 = self.last_faces_1
        self.last_faces_1 = self.current_faces
        faces = self.face_cascade.detectMultiScale(gray,self.scaling,self.neighbors,self.flags,self.min_size)
        self.current_faces = faces
        
    def visualizeCurrentFaces(self,frame):
        i=1
        for face in self.filter_by_persistence():
            x,y,w,h = face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),6,1)
            cv2.putText(frame, str(i), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
            i+=1
        return frame
    
    def _filter_by_persistence(self,current,last):
        possible_correspondence = {}
        for i in xrange(len(current)):
            xi,yi,wi,hi = current[i]
            sizei = wi*hi
            xi = xi+wi/2
            yi = yi+hi/2
            for j in xrange(len(last)):
                xj,yj,wj,hj = last[j]
                sizej = wj*hj
                xj = xj+wj/2
                yj = yj+hj/2
                if abs(sizei-sizej) / sizei > 0.25 or abs(xi - xj) / wi > 0.5 or abs(yi - yj) / hi > 0.5:
                    continue
                possible_correspondence[i] = j
        return [current[i] for i in possible_correspondence]

    def filter_by_persistence(self):
        last = self.last_faces_1
        last = self._filter_by_persistence(last, self.last_faces_2)
        current = self._filter_by_persistence(self.current_faces, last)
        return current
    
    def getCurrentFaces(self,frame):
        output = []
        active_faces = []
        if self.use_static:
            active_faces = self.current_faces
        else:
            active_faces = self.filter_by_persistence()
        for face in active_faces:
            x0,y0,w,h = face
            x0 += int(w*self.base_sub_region[0])
            y0 += int(h*self.base_sub_region[1])
            w = int(w*self.base_sub_region[2])
            h = int(h*self.base_sub_region[3])
            xf = x0+w
            yf = y0+h
            if x0 < 0:
                x0 = 0
            if xf > frame.shape[1]:
                xf = frame.shape[1]
            if y0 < 0:
                y0 = 0
            if yf > frame.shape[0]:
                yf = frame.shape[0]
            face = frame[y0:yf,x0:xf].copy()
            output.append(face)
        return output
    
    def run(self,frame):      
        frame = self.preprocessing_flow.execute(frame)
        self.setCurrentFaces(frame)
        output = self.getCurrentFaces(frame)
        if self.face_workflow:
            output = [face for face in output if self.face_workflow.execute(face)['face'][0]>0]
        return output
            
def extract_faces_from_directories(indirs,outdir):
    fd = OpenCVFaceDetection(use_static = True)
    if outdir[-1]!='/':
        outdir[-1] += '/'
    nfaces = 0
    nfiles = 0
    t0 = time.time()
    for d in indirs:
        files = os.listdir(d)
        if d[-1] != '/':
            d += '/'
        outcat = d[:-1].rsplit('/')[-1]
        for f in files:
            nfiles += 1     
            frame = cv2.imread(d+f)
            faces = fd.run(frame)
            if len(faces):
                for i in xrange(len(faces)):
                    nfaces += 1
                    face = faces[i]
                    fout = outdir + outcat + '_' + f + '_%d'%i + '.png'
                    cv2.imwrite(fout,face)
            if nfiles % 100 == 0:
                print 'processed',nfiles,'files, found',nfaces,'faces\ttime: ',time.time()-t0
    print 'done saving faces to',outdir
    
if __name__ == "__main__":
    import sys
    outdir = sys.argv[1]
    indirs = sys.argv[2:]
    extract_faces_from_directories(indirs,outdir)
    