import mdp_wrapper
import mdp
import numpy as np
from sklearn import svm
import time
from collections import defaultdict
from tools import classification_tools

class OneClassSVM(mdp.nodes.OneClassSVMScikitsLearnNode):
    
    def __init__(self, *args, **kwargs):
        super(OneClassSVM,self).__init__(*args,**kwargs)
        
    def is_supervised(self):
        return False

class SVC(mdp.nodes.SVCScikitsLearnNode):
    
    def __init__(self, *args, **kwargs):
        super(SVC,self).__init__(*args,**kwargs)
        
    def is_supervised(self):
        return True
    
    def execute(self,x):
        return self.label(x)
    
class AutoOptimizeSVC(mdp.Node):
    SVC = None
    data = None
    labels=None
    kwargs = None
    tuned_parameters = None
    
    def __init__(self,tuned_parameters,**kwargs):
        super(AutoOptimizeSVC,self).__init__()
        self.kwargs = kwargs
        self.tuned_parameters = tuned_parameters
        
    def is_supervised(self):
        return True
    
    def _train(self,data,labels):
        if not self.data:
            self.data = data.copy()
            self.labels = labels.copy()
        else:
            self.data = np.concatenate((self.data,data))
            self.labels = np.concatenate((self.labels,labels))
    
    def _stop_training(self):
        self.t0=time.time()
        self.labels = self.labels.flatten()
        optimizer = classification_tools.ParameterOptimizer(self.data,self.labels,mode='precision')
        self.SVC = optimizer.optimize_parameters(svm.SVC, self.tuned_parameters, **self.kwargs)
        del self.data
        del self.labels
        print "SVC Fit Time: ",time.time()-self.t0
        
    def _execute(self,x):
        return self.SVC.predict(x)
    
    def prob(self,x):
        return self.SVC.predict_proba(x)
        
class AutoOptimizeSVR(mdp.Node):
    SVR = None
    data = None
    labels=None
    norm_labels = False
    label_mean = None
    label_std = None
    kwargs = None
    tuned_parameters = None
    
    def __init__(self,tuned_parameters,norm_labels = False, **kwargs):
        super(AutoOptimizeSVR,self).__init__()
        self.kwargs = kwargs
        self.tuned_parameters = tuned_parameters
        self.norm_labels = norm_labels
        
    def is_supervised(self):
        return True
    
    def _train(self,data,labels):
        if not self.data:
            self.data = data.copy()
            self.labels = labels.copy()
        else:
            self.data = np.concatenate((self.data,data))
            self.labels = np.concatenate((self.labels,labels))
    
    def _stop_training(self):
        self.t0=time.time()
        self.labels = self.labels.flatten().astype(np.float64)
        if self.norm_labels:
            self.label_mean = np.mean(self.labels)
            self.labels -= self.label_mean
            self.label_std = np.std(self.labels)
            assert(self.label_std>0)
            self.labels /= self.label_std
            print 'label mean: ',self.label_mean,'\tlabel std: ',self.label_std
        optimizer = classification_tools.ParameterOptimizer(self.data,self.labels,mode='mse')
        self.SVR = optimizer.optimize_parameters(svm.SVR, self.tuned_parameters, **self.kwargs)
        del self.data
        del self.labels
        print "SVC Fit Time: ",time.time()-self.t0
        
    def _execute(self,x):
        x = self.SVR.predict(x)
        if self.norm_labels:
            x *= self.label_std
            x += self.label_mean
        return x

    
class SVR(mdp.nodes.SVRScikitsLearnNode):
    
    def __init__(self, *args, **kwargs):
        super(SVR,self).__init__(*args,**kwargs)
    
    def is_supervised(self):
        return True
    
    def execute(self,x):
        return self.label(x)
    
class VoteClassifier(mdp_wrapper.ListInputNode):
    max_class = 6
    
    def __init__(self, max_class = 6):
        super(VoteClassifier, self).__init__()
        self.max_class = max_class
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _execute(self,x):
        counts=defaultdict(int)
        for i in x:
            if i<0:
                i=0
            elif i>self.max_class:
                i=self.max_class
            counts[int(i)]+=1
        total=len(x)
        counts=counts.items()
        counts.sort(key=lambda x:x[1],reverse=True)
        confidence = counts[0][1]*1.0/total
        return counts[0][0],confidence
    
class SimpleAverageClassifier(mdp_wrapper.ListInputNode):
    max_class = 6
    
    def __init__(self, max_class = 6):
        self.max_class = max_class
        super(SimpleAverageClassifier, self).__init__()
    
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _execute(self,x):
        x = np.array(x)
        ret = x.mean()
        if ret<0:
            ret=0
        elif ret>self.max_class:
            ret=self.max_class
        return int(np.around(ret)),x.std()
    
