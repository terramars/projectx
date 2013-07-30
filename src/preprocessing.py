import mdp
import numpy as np
import cv2

class PreprocessingNode(mdp.Node):
    
    def __init__(self,*args,**kwargs):
        super(PreprocessingNode,self).__init__(*args,**kwargs)
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False

class ConvertToFloat(PreprocessingNode):
    
    def _execute(self,x):
        out = x.astype(np.float32)
        if x.dtype == np.uint8:
            out /= 255.0
        return out
    
    def execute(self,x):
        assert(type(x) == np.ndarray)
        return self._execute(x)
    
class ConvertToUInt8(PreprocessingNode):
    
    def _execute(self,x):
        x = x - x.min()
        if x.max():
            x = x * (255.5 / x.max())
        x = x.astype(np.uint8)
        return x
    
    def execute(self,x):
        assert(type(x) == np.ndarray)
        return self._execute(x)

class Grayscale(PreprocessingNode):
    
    def _execute(self,x):
        if len(x.shape)==2:
            return x
        elif x.shape[2] > 2:
            return cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
        else:
            assert(0)
    
    def execute(self,x):
        assert(type(x) == np.ndarray and x.dtype == np.uint8)
        return self._execute(x)
    
class GammaCorrection(PreprocessingNode):
    gamma = 0.2
    
    def __init__(self, gamma = 0.2):
        super(GammaCorrection,self).__init__()
        self.gamma = gamma
        assert(gamma >= 0 and gamma <= 1)
    
    def _execute(self,x):
        x = x.astype(np.float32)
        if self.gamma:
            return cv2.pow(x,self.gamma)
        else:
            return np.log(x)
    
    def execute(self,x):
        assert(type(x) == np.ndarray)
        return self._execute(x)
    
class ContrastEQ(PreprocessingNode):
    alpha = 0.1
    tau = 10.0
    
    def __init__(self, alpha = 0.1, tau = 10.0):
        super(ContrastEQ,self).__init__()
        self.alpha = alpha
        self.tau = tau
    
    def _execute(self,x):
        x = x.astype(np.float32)
        x = x / ( np.mean( cv2.pow(np.abs(x), self.alpha) ) ** (1/self.alpha) + 1e-6)
        absx = np.abs(x)
        x = x / ( np.mean( cv2.pow(np.choose(absx > self.tau,(self.tau,absx)), self.alpha ) ) ** (1/self.alpha) + 1e-6 )
        x = self.tau * np.tanh( x / self.tau )
        return x
    
    def execute(self,x):
        assert(type(x) == np.ndarray)
        return self._execute(x)    
    
class HistogramEQ(PreprocessingNode):
    
    def _execute(self,x):
        return cv2.equalizeHist(x)
    
    def execute(self,x):
        assert(type(x) == np.ndarray and x.dtype == np.uint8)
        return self._execute(x)
    
    
class PreprocessingFlow(mdp.Flow):
    
    def __init__(self,*args,**kwargs):
        super(PreprocessingFlow,self).__init__(*args,**kwargs)
    
    def is_trainable(self):
        return False
        
    def train(self,data_iterables):
        print "preprocessing flows cannot be trained"
        assert(0)
    
        
    
    
    
    