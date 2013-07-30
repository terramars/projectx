import mdp
import time
import sklearn.pls
import numpy as np

class Normalize(mdp.nodes.NormalizeNode):
    
    def __init__(self):
        super(Normalize,self).__init__(dtype='float64')
    
    def is_supervised(self):
        return False

class PCA(mdp.nodes.PCANode):
     
    def __init__(self, output_dim = 0, variance_kept = 0.95, **kwargs):
        super(PCA,self).__init__(**kwargs)
        if output_dim > 0:
            self.set_output_dim(output_dim)
        else:
            self.set_output_dim(variance_kept)
    
    def is_supervised(self):
        return False
    
    def train(self,data,print_results = True):
        t0=time.time()
        super(PCA,self).train(data)
        self.stop_training()
        if print_results:
            print "PCA Output Dimensionality:",self.output_dim,"\tExplained Variance:",self.explained_variance,"\tTime: ",time.time()-t0

class PCANIPALS(mdp.nodes.NIPALSNode):
     
    def __init__(self, output_dim = 0, variance_kept = 0.95, **kwargs):
        super(PCANIPALS,self).__init__(**kwargs)
        if output_dim > 0:
            self.set_output_dim(output_dim)
        else:
            self.set_output_dim(variance_kept)
    
    def is_supervised(self):
        return False
    
    def train(self,data,print_results = True):
        t0=time.time()
        super(PCANIPALS,self).train(data)
        self.stop_training()
        if print_results:
            print "PCA Output Dimensionality:",self.output_dim,"\tExplained Variance:",self.explained_variance,"\tTime: ",time.time()-t0
    
class FDA(mdp.nodes.FDANode):
    t0 = time.time()
    
    def __init__(self, output_dim = 10, **kwargs):
        super(FDA,self).__init__(**kwargs)
        self.set_output_dim(output_dim)
    
    def is_supervised(self):
        return True
    
    def train(self,data,labels):
        self.t0=time.time()
        if len(labels.shape)>1:
            assert(labels.shape[0]/labels.size==1)
        labels = labels.flatten()
        super(FDA,self).train(data,labels)
        
    def stop_training(self,print_results = True):
        super(FDA,self).stop_training()
        if print_results:
            print "FDA Output Dimensionality:",self.output_dim,"\tTime: ",time.time()-self.t0
            
            
class PLSRegression(mdp.Node):
    t0 = time.time()
    pls = None
    data = None
    labels = None
    label_mean = None
    label_std = None
    norm_labels = False
    
    def __init__(self, output_dim = 40, norm_labels = True, **kwargs):
        super(PLSRegression,self).__init__(output_dim = output_dim,**kwargs)
        self.pls = sklearn.pls.PLSRegression(n_components = output_dim)
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
        
    def _stop_training(self,print_results = True):
        self.t0=time.time()
        if self.norm_labels:
            self.label_mean = self.labels.mean(axis=0)
            self.labels -= self.label_mean
            self.label_std = self.labels.std(axis=0)
            self.label_std = np.choose(self.label_std>0,(1,self.label_std))
            self.labels /= self.label_std
            print 'label mean: ',self.label_mean,'\tlabel std: ',self.label_std
        self.pls.fit(self.data,self.labels)
        del self.data
        del self.labels
        if print_results:
            print "PLS Fit Time: ",time.time()-self.t0
            
    def _execute(self,x):
        return self.pls.transform(x)
    
### use this class for dimensionality reduction and/or nonlinear expansion    
class ManifoldFlow(mdp.Flow):
    
    def __init__(self,*args,**kwargs):
        super(ManifoldFlow,self).__init__(*args,**kwargs)
                
    def get_training_flags(self):
        training_flags = []
        for node in self:
            if node.is_trainable():
                if node.is_supervised():
                    training_flags.append(2)
                else:
                    training_flags.append(1)
            else:
                training_flags.append(0)
        return training_flags
                
    def is_trainable(self):
        return True
        
        
if __name__=="__main__":
    import numpy as np
    stuff = np.random.random((5,5))
    labels = np.ones((5,1))
    labels[:3]*=0
    print stuff,labels
    a=PLSRegression(output_dim = 3)
    a.train(stuff,labels)
    a.stop_training()
    stuff2 = np.random.random((1,5))
    proj = a.execute(stuff2)
    print stuff2,proj
            
    