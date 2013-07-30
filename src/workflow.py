import cPickle as cp
import os
import manifold
import time
import numpy as np
from tools import classification_tools, release

class Workflow(object):
    flow_graph      = None
    entry_point     = None
    features        = None
    exit_nodes      = None
    disabled_nodes  = None
    filename        = ''
    use_release = False
    
    def __init__(self, filename = ''):
        self.filename = filename
        self.flow_graph = {}
        self.features = {}
        self.exit_nodes = {}
        self.disabled_nodes = {}
        if filename and os.path.isfile(filename):
            self.load()
            
    def load(self):
        f = open(self.filename,'rb')
        print 'loading',self.filename
        t0=time.time()
        instr = f.read()
        try:
            tmp_dict = cp.loads(instr)
        except:
            instr = release.apply_key(instr)
            tmp_dict = cp.loads(instr)
        f.close()          
        self.__dict__.update(tmp_dict) 
        print 'done loading',self.filename,time.time()-t0

    def save(self):
        f = open(self.filename,'wb')
        print 'saving',self.filename
        t0=time.time()
        out = cp.dumps(self.__dict__,cp.HIGHEST_PROTOCOL)
        if self.use_release:
            out = release.apply_key(out)
        f.write(out)
        print 'done saving',self.filename,time.time()-t0
        f.close()
    
    def set_entry_point(self,entry_name,entry_flow):
        self.entry_point = entry_name
        self.flow_graph[entry_name] = (entry_flow,[],None)
        
    def add_flow(self,flow_name,parent_name,flow):
        self.flow_graph[flow_name] = (flow,[],parent_name)
        if flow_name not in self.flow_graph[parent_name][1]:
            self.flow_graph[parent_name][1].append(flow_name)
            
    def delete_flow(self,flow_name):
        if flow_name in self.exit_nodes:
            del self.exit_nodes[flow_name]
        if flow_name in self.flow_graph:
            _,children,parent = self.flow_graph[flow_name]
            if parent:
                pflow,pchild,pparent = self.flow_graph[parent]
                pchild = [i for i in pchild if i!=flow_name]
                self.flow_graph[parent] = (pflow,pchild,pparent)
                del self.flow_graph[flow_name]
                while len(children):
                    current = children[0]
                    if current in self.exit_nodes:
                        del self.exit_nodes[current]
                    if current not in self.flow_graph:
                        children = children[1:]
                        continue
                    _,subchildren,_ = self.flow_graph[current]
                    children = children [1:] + subchildren
                    del self.flow_graph[current]
        
    def add_exit_flow(self,flow_name,parent_name,feature_name,flow,mode='execute'):
        self.add_flow(flow_name,parent_name,flow)
        self.exit_nodes[flow_name] = (feature_name,mode)
        
    def add_feature(self,feature_name,output_processing_node):
        self.features[feature_name] = (output_processing_node,[])
        
    def execute_recurse(self,data,nodes):
        for node_name in nodes:
            if node_name in self.disabled_nodes:
                continue
            node,next_nodes,_ = self.flow_graph[node_name]
            if node_name in self.exit_nodes:
                feature_name,mode = self.exit_nodes[node_name]
                answer = -1
                if mode == 'prob':
                    answer = node.prob(data)[0][1]
                else:
                    answer = node.execute(data).flatten()[0]
                self.features[feature_name][1].append(answer)
            else:
                new_data = node.execute(data)
                self.execute_recurse(new_data,next_nodes)
        
    def execute(self,im):
        first_node,nodes,_ = self.flow_graph[self.entry_point]
        im = first_node.execute(im)
        self.execute_recurse(im, nodes)
        output = {}
        for feature_name,stuff in self.features.items():
            node,values = stuff
            if node:
                output[feature_name] = node.execute(values)
            else:
                output[feature_name] = values[0],1
        return output
    
    def train(self,data,labels,feature_name,exit_node = None,skip_end = False):
        to_train = []
        t_total = time.time()
        for flow_name in self.exit_nodes:
            if self.exit_nodes[flow_name][0]==feature_name and (not exit_node or flow_name == exit_node):
                current,_,parent_name=self.flow_graph[flow_name]
                to_train.append([(current,flow_name)])
                while parent_name:
                    flow_name = parent_name
                    current,_,parent_name = self.flow_graph[parent_name]
                    to_train[-1].insert(0,(current,flow_name))
        print 'training',len(to_train),'flows for feature',feature_name
        for complete_flow in to_train:
            working_data = [i.copy() for i in data]
            first_trainable = True
            for sub_flow,flow_name in complete_flow:
                if skip_end and flow_name in self.exit_nodes:
                    continue
                print 'started training',flow_name
                t0 = time.time()
                if sub_flow.is_trainable():
                    if first_trainable:
                        first_trainable = False
                        working_data = np.array(working_data)
                        working_data = working_data.reshape((working_data.shape[0],working_data.shape[2]))
                    if isinstance(sub_flow,manifold.ManifoldFlow):
                        if sub_flow[-1].get_remaining_train_phase():
                            training_flags = sub_flow.get_training_flags()
                            training_input = []
                            for flag in training_flags:
                                if flag==0:
                                    training_input.append(None)
                                elif flag==1:
                                    training_input.append([working_data])
                                elif flag==2:
                                    training_input.append([(working_data,labels)])
                                else:
                                    print "ERROR: invalid training flag",flag
                                    assert(0)
                            sub_flow.train(training_input)
                    elif sub_flow.get_remaining_train_phase():
                        if sub_flow.is_supervised():
                            sub_flow.train(working_data,labels)
                            sub_flow.stop_training()
                        else:
                            sub_flow.train(working_data)
                            sub_flow.stop_training()
                    
                    working_data = sub_flow.execute(working_data)
                elif first_trainable:
                    for i in xrange(len(working_data)):
                        working_data[i] = sub_flow.execute(working_data[i])
                else:
                    working_data = sub_flow.execute(working_data)
                print 'done training flow',flow_name,'\t',time.time()-t0
            if not skip_end:
                if len(working_data.shape)==1:
                    working_data = working_data.reshape((working_data.shape[0],1))
                print 'training error: ',np.sum(np.abs(working_data-labels))  ### for some reason i'm not sure wtf is going on with regards to this number
            self.save()
        print 'done training feature',feature_name,'\t',time.time()-t_total
                
def test_workflow(model,ftest,feature,mode='class', thresh = None):
    import cv2
    from sklearn.metrics import confusion_matrix
    t0=time.time()
    i=0
    y_true = []
    y_pred = []
    for line in open(ftest):
        i+=1
        if i%100==0:
            print i,time.time()-t0
        line = line.strip().split()
        im = cv2.imread(line[0])
        for tmp_feature in model.features:
            del model.features[tmp_feature][1][:]
        output = model.execute(im)
        predict = output[feature][0]
        y_true.append(int(line[1]))
        y_pred.append(predict)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if mode == 'class':
        y_pred = [int(np.round(i)) for i in y_pred]
        print classification_tools.classification_report(y_true, y_pred)
        print confusion_matrix(y_true,y_pred)
        print
    else:
        print classification_tools.regression_report(y_true,y_pred)
        if thresh:
            for i in range(len(thresh)):
                idx = [j for j in range(len(y_true)) if y_true[j]<thresh[i][1] and y_true[j]>=thresh[i][0]]
                tmp_y_true = [y_true[j] for j in idx]
                tmp_y_pred = [y_pred[j] for j in idx]
                print 'thresh: ',thresh[i]
                print classification_tools.regression_report(tmp_y_true,tmp_y_pred)
            y_pred_class = []
            for y in y_pred:
                for j in range(len(thresh)):
                    if y < thresh[j][1]:
                        y_pred_class.append(j)
                        break
            y_true_class = []
            for y in y_true:
                for j in range(len(thresh)):
                    if y<thresh[j][1]:
                        y_true_class.append(j)
                        break
            print classification_tools.classification_report(y_true_class,y_pred_class)
            print confusion_matrix(y_true_class,y_pred_class)
    
def execute_untrainable_flow_on_list(data,flow):
    for i in xrange(len(data)):
        data[i] = flow.execute(data[i])
    return data

def init_age_classifiers(model):
    import classifier
    tuned_parameters_rbf = [{'gamma': [0.02, 0.01, 0.005, 0.001],
                       'C': [0.1, 1, 10],
                       'epsilon': [0.4, 0.2, 0.1, 0.05]}]
    clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVRAR1","NormA1","age",clas)
#     clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
#     model.add_exit_flow("SVRAR2","NormA2","age",clas)
#     clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
#     model.add_exit_flow("SVRAR3","NormA3","age",clas)
    clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVRAR4","NormA4","age",clas)
    clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVRAR5","NormA5","age",clas)
    clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVRAR6","NormA6","age",clas)
    clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVRAR7","NormA7","age",clas)
    clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVRAR8","NormA8","age",clas)
#     clas = classifier.AutoOptimizeSVR(tuned_parameters_rbf, norm_labels = True, kernel = 'rbf', tol = 1e-6)
#     model.add_exit_flow("SVRAR9","NormA9","age",clas)
#     tuned_parameters_rbf = [{'gamma': [0.02, 0.01, 0.005, 0.001],
#                        'C': [0.01, 0.1, 1, 10, 100, 1000]}]
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', class_weight='auto')
#     model.add_exit_flow("SVCAR1","NormA1","age",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', class_weight='auto')
#     model.add_exit_flow("SVCAR2","NormA2","age",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', class_weight='auto')
#     model.add_exit_flow("SVCAR3","NormA3","age",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', class_weight='auto')
#     model.add_exit_flow("SVCAR4","NormA4","age",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', class_weight='auto')
#     model.add_exit_flow("SVCAR5","NormA4","age",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', class_weight='auto')
#     model.add_exit_flow("SVCAR6","NormA4","age",clas)
    print model.__dict__
    model.save()
    
def init_gender_classifiers(model):
    import classifier
    tuned_parameters_rbf = [{'gamma': [0.04, 0.02, 0.01, 0.005],
                       'C': [0.1, 1, 10, 100]}]
    clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVCGR1","NormG1","gender",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
#     model.add_exit_flow("SVCGR2","NormG2","gender",clas)
    clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVCGR3","NormG3","gender",clas)
    clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVCGR4","NormG4","gender",clas)
    clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVCGR5","NormG5","gender",clas)
    clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
    model.add_exit_flow("SVCGR6","NormG6","gender",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
#     model.add_exit_flow("SVCGR7","NormG7","gender",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
#     model.add_exit_flow("SVCGR8","NormG8","gender",clas)
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', tol = 1e-6)
#     model.add_exit_flow("SVCGR9","NormG9","gender",clas)
    model.save()

if __name__ == '__main__':
#     import classifier
#     model = Workflow('../models/age_gender_v0.1')
#     model = Workflow('../models/smile_v0.0')
#     print model.__dict__
#     model.use_release = True
#     model.filename = '../models/age_gender_release_v0.1'
#     model.save()
#     model.exit_nodes['SVCSm']=('smile','prob')
#     model.save()
#     model.delete_flow('Norm7a')
#     model.delete_flow('Norm8a')
#     model.delete_flow('PixHOGOLBP150')
#     model.delete_flow('PLSG9')
#     model.delete_flow('PLSG8')
#     model.delete_flow('NormA9a')
#     model.delete_flow('NormA8a')
#     model.delete_flow('NormA7a')
#     model.disabled_nodes = {}
#     model.add_feature('glasses',None)
#     model.add_feature('age',classifier.SimpleAverageClassifier(100))
#     model.save()

#     init_age_classifiers(model)
#     from tools import data_tools
#     from random import shuffle
#     tuned_parameters_rbf = [{'gamma': [0.04, 0.02],
#                              'C': [1, 10, 100]}]
# #     tuned_parameters_linear = [{'C':[0.01,0.1,1,10,100]}]
#     clas = classifier.AutoOptimizeSVC(tuned_parameters_rbf,kernel = 'rbf', probability = True, tol = 1e-6)
#     model.add_exit_flow("SVCSm","NormSm","smile",clas, mode = 'prob')
# #     data,labels = data_tools.load_label_file_to_data('/home/elliot/projectx/data/faces/age/agegender-pls-train.txt')
# #     data,labels = data_tools.load_label_file_to_data('/home/elliot/projectx/data/Japanese-Celebrities/train-pls-all.txt')
#     data,labels = data_tools.load_label_file_to_data('/home/elliot/projectx/data/glasses/pls.txt')
#     rands = list(range(len(data)))
#     shuffle(rands)
#     data = [data[i] for i in rands]
#     labels = np.array([labels[i,:] for i in rands],dtype=np.float64)
#     model.train(data,labels,"glasses",skip_end = True, exit_node = 'SVCGl')
# #     model.train(data,labels,"age",skip_end = True, exit_node = 'SVRAR8')
# #     model.train(data,labels,"age",skip_end = True, exit_node = 'SVRAR9')
#     data,labels = data_tools.load_label_file_to_data('/home/elliot/projectx/data/faces/gender/agegender-svm-train-gender.txt')
#     data,labels = data_tools.load_label_file_to_data('/home/elliot/projectx/data/Japanese-Celebrities/train-svm-all-age.txt')
#     data,labels = data_tools.load_label_file_to_data('/home/elliot/projectx/data/smile/train-svm.txt')
#     rands = list(range(len(data)))
#     shuffle(rands)
#     data = [data[i] for i in rands]
#     labels = np.array([labels[i,:] for i in rands],dtype=np.float64)
#     model.train(data,labels,"age",skip_end = False)
# #     model.train(data,labels,"age",skip_end = False, exit_node = 'SVRAR8')
# #     model.train(data,labels,"age",skip_end = False, exit_node = 'SVRAR9')

#     model.disabled_nodes['SVCGR1']=1
#     model.disabled_nodes['SVCGR2']=1
#     model.disabled_nodes['SVCGR3']=1
#     model.disabled_nodes['SVCGR4']=1
#     model.disabled_nodes['SVCGR5']=1
#     model.disabled_nodes['SVCGR6']=1
#     model.disabled_nodes['SVCGR7']=1
#     model.disabled_nodes['SVCGR8']=1
#     model.disabled_nodes['SVCGR9']=1
#     del model.disabled_nodes['PLSG2']
#     model.disabled_nodes['SVRAR1']=1
#     model.disabled_nodes['PIX150']=1
#     model.disabled_nodes['PLSA3']=1
#     model.disabled_nodes['SVRAR2']=1
#     model.disabled_nodes['SVRAR3']=1
#     model.save()
#     model.disabled_nodes['SVRAR4']=1
#     model.disabled_nodes['SVRAR5']=1
#     model.disabled_nodes['SVRAR6']=1
#     
#     model.disabled_nodes['SVRAR7']=1
#     model.disabled_nodes['SVRAR8']=1
#     model.disabled_nodes['SVRAR9']=1
     
#     test_workflow(model,'/home/elliot/projectx/data/smile/test.txt','smile', mode='class')
#     test_workflow(model,'/home/elliot/projectx/data/faces/gender/agegender-test-2-gender.txt','gender', mode='class')
#     test_workflow(model,'/home/elliot/projectx/data/Japanese-Celebrities/test-age.txt','age', mode='reg', thresh=[(-1,12),(12,20),(20,35),(35,55),(55,1000)])
#     model.disabled_nodes['SVRAR7'] = 1
#     del model.disabled_nodes['SVRAR8']
#     test_workflow(model,'/home/elliot/projectx/data/Japanese-Celebrities/test-age.txt','age', mode='reg', thresh=[(-1,12),(12,20),(20,35),(35,55),(55,1000)])
#     model.disabled_nodes['SVRAR8'] = 1
#     del model.disabled_nodes['SVRAR9']
#     test_workflow(model,'/home/elliot/projectx/data/Japanese-Celebrities/test-age.txt','age', mode='reg', thresh=[(-1,12),(12,20),(20,35),(35,55),(55,1000)])
#     model.disabled_nodes['SVRAR3'] = 1
#     del model.disabled_nodes['SVRAR4']
#     test_workflow(model,'/home/elliot/projectx/data/Japanese-Celebrities/test-age.txt','age', mode='reg', thresh=[(-1,12),(12,20),(20,35),(35,55),(55,1000)])
#     model.disabled_nodes['SVRAR4'] = 1
#     del model.disabled_nodes['SVRAR5']
#     test_workflow(model,'/home/elliot/projectx/data/Japanese-Celebrities/test-age.txt','age', mode='reg', thresh=[(-1,12),(12,20),(20,35),(35,55),(55,1000)])
#     model.disabled_nodes['SVRAR5'] = 1
#     del model.disabled_nodes['SVRAR6']
#     test_workflow(model,'/home/elliot/projectx/data/Japanese-Celebrities/test-age.txt','age', mode='reg', thresh=[(-1,12),(12,20),(20,35),(35,55),(55,1000)])
    
#     import preprocessing
#     import filters
#     import descriptor
#     pre = preprocessing.PreprocessingFlow([preprocessing.Grayscale(), preprocessing.HistogramEQ(), preprocessing.ContrastEQ(), preprocessing.ConvertToUInt8()])
#     model.set_entry_point("GrayHistEQConEQ",pre)
#     filt = filters.FilterFlow( [ filters.SubRegion([(0.2,0.2,0.8,0.6)]) ] )
#     model.add_flow("Sub", "GrayHistEQConEQ", filt)
    
#     filt = filters.FilterFlow( [ filters.Resize( [(150,75)] ) ] )
#     model.add_flow("ResSm","Sub",filt)
#     filt = filters.FilterFlow( [ filters.Resize( [(120,120),(60,60),(30,30)] ) ] )
#     model.add_flow("Res1206030","Sub",filt)
#     filt = filters.FilterFlow( [ filters.OLBP('uniform') ] )
#     model.add_flow("OLBP150","Res150",filt)
#     model.add_flow("OLBP1206030","Res1206030",filt)
#     des = descriptor.Pixels()
#     model.add_flow("Pix150","Res150",des)
#     model.add_flow("Pix1206030","Res1206030",des)
#     des = descriptor.HOG(win_size=(96,64))
#     model.add_flow("HOG","Sub",des)
#     model.add_flow("HOG1206030","Res1206030",des)
#     des = descriptor.SpatialHistogram(size = (8,8), bins = 59)
#     model.add_flow('SH150','OLBP150',des)
#     filt = filters.FilterFlow( [ filters.OLBP(), descriptor.SpatialHistogram(size = (6,6), bins = 59) ] )
#     des = descriptor.ConcatenatingFlow( [  filt, descriptor.HOG(win_size = (96,96)) ] )
#     model.add_flow('HOGOLBP1206030','Res1206030',des)
#     des = descriptor.ConcatenatingFlow( [ filt, descriptor.HOG(win_size = (96,96)) ] )
#     model.add_flow('HOGOLBP150','Res150',des)
#     des = descriptor.ConcatenatingFlow( [ filt, descriptor.HOG(win_size = (96,96)), descriptor.Pixels() ] )
#     model.add_flow('PixHOGOLBP150','Res150',des)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSGl","HOG",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSG8","HOGOLBP1206030",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSG9","PixHOGOLBP150",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSA7","HOGOLBP150",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSA8","HOGOLBP1206030",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSA9","PixHOGOLBP150",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSA4","HOG1206030",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSA5","SH150",mani)
#     mani = manifold.ManifoldFlow( [ manifold.PLSRegression(dtype = 'float64', output_dim = 40) ] )
#     model.add_flow("PLSA6","SH1206030",mani)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormGl', 'PLSGl', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormG8', 'PLSG8', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormG9', 'PLSG9', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormA7', 'PLSA7', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormA8', 'PLSA8', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormA9', 'PLSA9', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormA4', 'PLSA4', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormA5', 'PLSA5', norm)
#     norm = manifold.ManifoldFlow([ manifold.Normalize() ] )
#     model.add_flow('NormA6', 'PLSA6', norm)
#     init_gender_classifiers(model)
#     init_age_classifiers(model)
#     print model.__dict__
#     model.save()
    pass
    
    
    