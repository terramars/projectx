import os
import sys
from random import sample
import cv2
import numpy as np
import time

def folders_to_label_file(dirpos, dirneg, fout):
    pos = os.listdir(dirpos)
    neg = os.listdir(dirneg)
    fout = open(fout,'w')
    for f in pos:
        fout.write(dirpos+f+" 1\n")
    for f in neg:
        fout.write(dirneg+f+" 0\n")
    fout.close()
        
def sample_label_file(fin,ftrain,ftest,ntrain,ntest):
    labels={}
    fin=open(fin)
    ftrain=open(ftrain,'w')
    ftest=open(ftest,'w')
    for line in fin:
        line=line.strip().split()
        f = line[0]
        label = tuple(line[1:])
        if len(label)==1:
            label = int(label[0])
        if label not in labels:
            labels[label]=[]
        labels[label].append(f)
    for label,files in labels.items():
        nlabel = ntrain
        if ntrain < 1:
            nlabel = int(ntrain * len(files))
        print label,len(files)
        if len(files)>nlabel:
            train = sample(files,nlabel)
        else:
            train = files
        for f in train:
            labelstr=''
            if type(label)==type(1):
                labelstr=str(label)
            else:
                for l in label:
                    labelstr+=str(l)+' '
                labelstr=labelstr[:-1]
            ftrain.write(f+" "+labelstr+"\n")
        files = [f for f in files if f not in train]
        test = files
        nlabel = ntest
        if ntest <= 1:
            nlabel = int(ntest * len(files))
        if len(files)>nlabel:
            test = sample(files,nlabel)
        for f in test:
            labelstr=''
            if type(label)==type(1):
                labelstr=str(label)
            else:
                for l in label:
                    labelstr+=str(l)+' '
                labelstr=labelstr[:-1]
            ftest.write(f+" "+labelstr+"\n")
    ftrain.close()
    ftest.close()
    
def load_label_file_to_data(flabel):
    data = []
    labels = []
    i=0
    t0=time.time()
    for line in open(flabel):
        i+=1
        if i%1000 == 0:
            print "loading: ",i,time.time()-t0
        line = line.strip().split()
        if not os.path.isfile(line[0]):
            continue
        data.append(cv2.imread(line[0]))
        if len(line)>2:
            labels.append(line[1:])
        else:
            labels.append([line[1]])
    print 'done loading images: ',i,time.time()-t0
    labels = np.array(labels,np.int32)
    labels = labels.astype(np.float64)
    return data,labels
        
if __name__ == '__main__':
#     dirpos = sys.argv[1]
#     dirneg = sys.argv[2]
#     fout = sys.argv[3]
#     folders_to_label_file(dirpos,dirneg,fout)
    fin = sys.argv[1]
    ftrain = sys.argv[2]
    ftest = sys.argv[3]
    ntrain = float(sys.argv[4])
    ntest = float(sys.argv[5])
    if ntrain>1:
        ntrain = int(ntrain)
    if ntest>1:
        ntest = int(ntest)
    sample_label_file(fin,ftrain,ftest,ntrain,ntest)