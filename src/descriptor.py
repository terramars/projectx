import mdp_wrapper
import mdp
import numpy as np
from scipy.misc import imresize
import cv2

class Pixels(mdp_wrapper.ListInputNode):
    
    def __init__(self):
        super(Pixels,self).__init__()
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _execute(self,x):
        x = [np.ndarray.flatten(im) for im in x]
        x = np.concatenate(x)
        return x.reshape((1,x.shape[0]))
    
class SpatialHistogram(mdp_wrapper.ListInputNode):
    size = None
    bins = None
    
    def __init__(self,size = (8,8), bins = 59):
        self.size = size
        self.bins = bins
        super(SpatialHistogram,self).__init__()
    
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _hist(self,X):
        height,width = X.shape
        rows,cols = self.size
        py = int(np.floor(height/rows))
        px = int(np.floor(width/cols))
        E = []
        for row in range(0,rows):
            for col in range(0,cols):
                C = X[row*py:(row+1)*py,col*px:(col+1)*px]
                H = np.bincount(C.flatten(),minlength = self.bins) / float(C.size)
                E.extend(H)
        return np.asarray(E)
    
    def _execute(self,x):
        x = [self._hist(im) for im in x]
        x = np.concatenate(x)
        return x.reshape((1,x.shape[0]))
        
    
class HOG(mdp_wrapper.ListInputNode):
    
    def __init__(self, win_size = (64,128), block_size = (16,16), block_stride = (8,8), cell_size = (8,8), nbins = 9):
        self.d = cv2.HOGDescriptor(_winSize = win_size, _blockSize = block_size, _blockStride = block_stride, _cellSize = cell_size, _nbins = nbins)
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        super(HOG,self).__init__()
        
    def __getstate__(self):
        result = self.__dict__.copy()
        del result['d']
        return result
    
    def __setstate__(self,dic):
        self.__dict__ = dic
        self.d = cv2.HOGDescriptor(_winSize = self.win_size, _blockSize = self.block_size, _blockStride = self.block_stride, _cellSize = self.cell_size, _nbins = self.nbins)
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _execute(self,x):
        x = [self.d.compute(imresize(im,(self.win_size[1],self.win_size[0]),'bilinear')) for im in x]
        x = np.concatenate(x)
        return x.reshape((1,x.shape[0]))
    
class ConcatenatingFlow(mdp.Flow):
    
    def __init__(self,*args,**kwargs):
        super(ConcatenatingFlow,self).__init__(*args,**kwargs)
    
    def is_trainable(self):
        return False
    
    def execute(self,x):
        out = None
        for node in self:
            x = node.execute(x)
            if not isinstance(out,np.ndarray):
                out = np.array(x)
            else:
                out = np.concatenate([out,x],axis=1)
        return out
    
    def train(self,data_iterables):
        print "concatenating flows cannot be trained"
        assert(0)
    