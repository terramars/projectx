import mdp_wrapper
import mdp
from scipy.misc import imresize
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class OLBP(mdp_wrapper.ListInputOutputNode):
    
    mode = None
    LUT = None
    
    def __init__(self, mode = 'uniform'):
        super(OLBP,self).__init__()
        self.mode = mode
        if mode == 'uniform':
            self.LUT = np.array([1,    2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
                                 12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
                                 17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                 18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
                                 23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                 24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                 25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
                                 30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
                                 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
                                 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
                                 37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
                                 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
                                 43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
                                 48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58],dtype=np.uint8)
        elif mode == 'rotation':
            self.LUT = np.array([1,   2,   2,   3,   2,   0,   3,   4,   2,   0,   0,   0,   3,   0,   4,   5,      #16
                                 2,   0,   0,   0,   0,   0,   0,   0,   3,   0,   0,   0,   4,   0,   5,   6,      #32
                                 2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,      #48
                                 3,   0,   0,   0,   0,   0,   0,   0,   4,   0,   0,   0,   5,   0,   6,   7,      #64
                                 2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,      #80
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,      #96
                                 3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,      #112
                                 4,   0,   0,   0,   0,   0,   0,   0,   5,   0,   0,   0,   6,   0,   7,   8,      #128
                                 2,   3,   0,   4,   0,   0,   0,   5,   0,   0,   0,   0,   0,   0,   0,   6,      #144
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   7,      #160
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,      #176
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,      #192
                                 3,   4,   0,   5,   0,   0,   0,   6,   0,   0,   0,   0,   0,   0,   0,   7,      #208
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,      #224
                                 4,   5,   0,   6,   0,   0,   0,   7,   0,   0,   0,   0,   0,   0,   0,   8,      #240
                                 5,   6,   0,   7,   0,   0,   0,   8,   6,   7,   0,   8,   7,   8,   8,   9],dtype=np.uint8)     #256
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _execute(self,x):
        X = np.asarray(x)
        X = (1<<7) * (X[0:-2,0:-2] >= X[1:-1,1:-1]) \
            + (1<<6) * (X[0:-2,1:-1] >= X[1:-1,1:-1]) \
            + (1<<5) * (X[0:-2,2:] >= X[1:-1,1:-1]) \
            + (1<<4) * (X[1:-1,2:] >= X[1:-1,1:-1]) \
            + (1<<3) * (X[2:,2:] >= X[1:-1,1:-1]) \
            + (1<<2) * (X[2:,1:-1] >= X[1:-1,1:-1]) \
            + (1<<1) * (X[2:,:-2] >= X[1:-1,1:-1]) \
            + (1<<0) * (X[1:-1,:-2] >= X[1:-1,1:-1])
        if isinstance(self.LUT,np.ndarray):
            X = np.take(self.LUT,X)
        return [X.astype(np.uint8)]

class SubRegion(mdp_wrapper.ListInputOutputNode):
    
    regions = None
    
    def __init__(self,regions = None):
        super(SubRegion,self).__init__()
        self.regions = regions
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _execute(self,x):
        subregions = []
        data = x
        for rect in self.regions:
            x0,y0,x1,y1 = rect
            if type(x0) == type(0.1):
                x0*=data.shape[0]
            if type(x1) == type(0.1):
                x1*=data.shape[0]
            if type(y0) == type(0.1):
                y0*=data.shape[1]
            if type(y1) == type(0.1):
                y1*=data.shape[1]
            subregions.append(data[y0:y1,x0:x1].copy())
#         import cv2
#         cv2.imshow('tmp',subregions[-1])
#         cv2.waitKey()
        return subregions
    
    
class Resize(mdp_wrapper.ListInputOutputNode):
    
    sizes = None
    mode = None
    
    def __init__(self, sizes, mode = None):
        super(Resize, self).__init__()
        if type(sizes) != type([]):
            sizes = [sizes]
        self.sizes = sizes
        if not mode:
            self.mode = 'bilinear'
        else:
            self.mode = mode
            
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
        
    def _execute(self,x):
        resized = []
        for size in self.sizes:
            resized.append(imresize(x, size, self.mode))
        return resized
    

### use this class to calculate image filters sequentially (can multiplex flow output in architecture module)    
class FilterFlow(mdp.Flow):
    output_dim = None
    
    def __init__(self,*args,**kwargs):
        super(FilterFlow,self).__init__(*args,**kwargs)
    
    def is_trainable(self):
        return False
    
    def execute(self,x):
        for node in self:
            x = node.execute(x)
        return x
    
    def train(self,data_iterables):
        print "filter flows cannot be trained"
        assert(0)
        
        