# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 22:39:05 2022

@author: yasmmin
"""

import os
import re
import urllib.request
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
h = .02  # step size in the mesh

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('always')  

from sklearn.cluster import KMeans

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

import statistics as st
from joblib import dump, load

class Implementation_vaxijen:
    descriptors={}
    
    def __init__(self):
        self.descriptors=self.build_descriptors()
    
    def build_descriptors(self):
        descriptors={}
        f=open("e_table_descriptors.tsv","r")
        for line in f:
            l=line.replace("\n","").split(" ")
            aa=l[0]
            zds=[]
            for z in l[1:]:
                zds.append(float(z))
            descriptors[aa]=zds
        f.close()
        
        return descriptors
    
    def preprocess_remove_newLine(self, fasta):
        seq=""
        id_=""
        fasta=fasta.replace("_new","")
        g=open(fasta.split(".")[0]+"_new."+fasta.split(".")[1],"w")
        f=open(fasta,"r")
        for line in f:
            l=line.replace("\n","")
            if(l.startswith(">")):
                if(seq!=""):
                    g.write("%s\n%s\n" %(id_, seq) )
                id_=l.replace('\t','_').replace('.','_')
                seq=""
            else:
                seq+=l.replace('X','').replace('Z','')
        
        if(seq!=""):
            g.write("%s\n%s\n" %(id_, seq) )
        
        g.close()
        f.close()
    
    def _calculate_features(self, seq, mode):
        aac={}
        
        n=len(seq)
        
        l=8
        lags=list(range(1, l+1))
        a=[]
        for l in lags:
            aac["l"+str(l)]=[]
            
            for i in range(5):
                for j in range(5):
                    if(mode=='auto'):
                        cond=(i==j)
                    if(mode=='cross'):
                        cond=(i!=j)
                        
                    if(cond):
                        s=0
                        
                        k=0
                        while k<(n-l):
                            aa=seq[k]
                            flag=False
                            if(aa in self.descriptors.keys()):
                                e=self.descriptors[aa]
                                flag=True
                            
                            aal=seq[k+l]
                            flag2=False
                            if(aal in self.descriptors.keys()):
                                el=self.descriptors[aal]
                                flag2=True
                            
                            if(flag and flag2):
                                s+= (e[j]*el[j]) / (n-l)
                            
                            k+=1
                        
                        aac["l"+str(l)].append(s)
        return aac                
                        
        
    def build_dataset_matrix_variance(self, mode, ide, method):
        fasta_pos=ide+"/"+method+"/dataset_pos.fasta"
        fasta_neg=ide+"/"+method+"/dataset_neg.fasta"
        
        classes={"pos": 1, "neg": 0}
        
        g=open(ide+"/"+method+"/"+mode+"_dataset.tsv", "w")
        feas=[]
        for i in range(1, 26):
            feas.append("feature"+str(i))
            
        g.write("protein\tclass\tlag\tfeatures\n")
        
        for cl in classes.keys():
            fasta=eval("fasta_"+cl)
            if(fasta!=""):
                class_=classes[cl]
                
                self.preprocess_remove_newLine(fasta)
                
                id_=""
                f=open(fasta.split(".")[0]+"_new."+fasta.split(".")[1], "r")
                for line in f:
                    l=line.replace("\n","")
                    if(not l.startswith(">")):
                        if(id_!=""):
                            data = eval("self._calculate_features(l, mode)")
                            for k in data.keys():
                                strar=[]
                                for e in data[k]:
                                    strar.append( "{:.5f}".format(e) )
                                    
                                features=','.join(strar)
                                g.write("%s\t%i\t%s\t%s\n" %(id_, class_, k, features) )
                    else:
                        try:
                            id_=l.replace(">","").split("|")[1].replace('\t','').replace('.','').replace('-','')
                        except:
                            try:
                                id_=l.replace(" ","").split("|")[0].replace('\t','').replace('.','').replace('-','')
                            except:
                                id_=""
                f.close()
            
        g.close()
        
class Implementation_vaxijenModified:
    descriptors={}
    
    def __init__(self):
        self.descriptors=self.build_descriptors()
    
    def build_descriptors(self):
        descriptors={}
        f=open("descriptors_pmc5549711.tsv","r")
        for line in f:
            l=line.replace("\n","").split("\t")
            aa=l[0]
            zds=[]
            for z in l[1:]:
                zds.append(float(z))
            descriptors[aa]=zds
        f.close()
        
        return descriptors
    
    def preprocess_remove_newLine(self, fasta):
        seq=""
        id_=""
        fasta=fasta.replace("_new","")
        g=open(fasta.split(".")[0]+"_new."+fasta.split(".")[1],"w")
        f=open(fasta,"r")
        for line in f:
            l=line.replace("\n","")
            if(l.startswith(">")):
                if(seq!=""):
                    g.write("%s\n%s\n" %(id_, seq) )
                id_=l
                seq=""
            else:
                seq+=l.replace('X','').replace('Z','')
        
        if(seq!=""):
            g.write("%s\n%s\n" %(id_, seq) )
        
        g.close()
        f.close()
    
    def build_fastas(self):
        i=0
        g=open("alternative_dataset/dataset_pos.fasta","w")
        f=open("alternative_dataset/positive_set.tsv","r")
        for line in f:
            l=line.replace("\n","").split('\t')
            if(i>0):
                g.write(">%s\n%s\n" %( l[0]+"_"+l[2], l[1] ) )
            i+=1
        f.close()
        g.close()
        
        i=0
        g=open("alternative_dataset/dataset_neg.fasta","w")
        f=open("alternative_dataset/negative_set.tsv","r")
        for line in f:
            l=line.replace("\n","").split('\t')
            if(i>0):
                g.write(">%s\n%s\n" %( l[0]+"_"+l[2], l[1] ) )
            i+=1
        f.close()
        g.close()
        
    def _get_optimal_lag(self, fasta_pos, fasta_neg):
        seqs=set()
        f=open(fasta_pos, "r")
        for line in f:
            l=line.replace("\n","")
            if(not l.startswith(">")):
                seqs.add(l)
        f.close()
        
        f=open(fasta_neg, "r")
        for line in f:
            l=line.replace("\n","")
            if(not l.startswith(">")):
                seqs.add(l)
        f.close()
        
        n=len(seqs)
        i=30
        init=1
        while(init!=n):
            init=0
            for s in seqs:
                if(len(s)>=i):
                    init+=1
            i-=1    
        if(i<4):
            i=4
        return i
    
    def _calculate_features(self, seq, max_lag):
        aac={}
        
        n=len(seq)
        
        l=max_lag
        
        lags=list(range(1, l+1))
        a=[]
        for l in lags:
            aac["l"+str(l)]=[]
            
            for j in range(6):
                mean_=0
                for aa in seq:
                    el=self.descriptors[aa]
                    mean_+=el[j]
                mean_=mean_/n
                
                s=0
                i=0
                while i<(n-l):
                    aa=seq[i]
                    flag=False
                    if(aa in self.descriptors.keys()):
                        e=self.descriptors[aa]
                        va=e[j] - mean_
                        flag=True
                    
                    aal=seq[i+l]
                    flag2=False
                    if(aal in self.descriptors.keys()):
                        el=self.descriptors[aal]
                        vb=el[j] - mean_
                        flag2=True
                    
                    if(flag and flag2):
                        s+= (va * vb) / (n-l)
                    
                    i+=1
                        
                aac["l"+str(l)].append(s)
        return aac                
                        
        
    def build_dataset_matrix_variance(self, mode, ide, method):
        fasta_pos=ide+"/"+method+"/dataset_pos.fasta"
        fasta_neg=ide+"/"+method+"/dataset_neg.fasta"
        
        max_lag=self._get_optimal_lag(fasta_pos, fasta_neg)
        print("max lag", max_lag)
        
        classes={"pos": 1, "neg": 0}
        
        g=open(ide+"/"+method+"/"+mode+"_dataset.tsv", "w")
        feas=[]
        for i in range(1, 7):
            feas.append("feature"+str(i))
            
        g.write("protein\tclass\tlag\tfeatures\n")
        
        for cl in classes.keys():
            fasta=eval("fasta_"+cl)
            if(fasta!=""):
                class_=classes[cl]
                
                self.preprocess_remove_newLine(fasta)
                
                id_=""
                f=open(fasta.split(".")[0]+"_new."+fasta.split(".")[1], "r")
                for line in f:
                    l=line.replace("\n","")
                    if(not l.startswith(">")):
                        if(id_!=""):
                            data = eval("self._calculate_features(l, max_lag)")
                            for k in data.keys():
                                strar=[]
                                for e in data[k]:
                                    strar.append( "{:.5f}".format(e) )
                                    
                                features=','.join(strar)
                                g.write("%s\t%i\t%s\t%s\n" %(id_, class_, k, features) )
                    else:
                        try:
                            id_=l.replace(">","").split("|")[1].replace('\t','').replace(' ','').replace('.','').replace('-','')
                        except:
                            try:
                                id_=l.replace(" ","").split("|")[0].replace('\t','').replace(' ','').replace('.','').replace('-','')
                            except:
                                id_=""
                f.close()
            
        g.close()

 
class Implementation_new_158:
    descriptors={}
    
    def __init__(self):
        self.descriptors=self.build_descriptors()
    
    def build_descriptors(self):
        descriptors={}
        df=pd.read_csv('filtered_aaindex.tsv', sep='\t')
        for k in df.columns:
            if(k!='id' and k!='name' and k!='description'):
                descriptors[k]=list(df[k])
        
        return descriptors
    
    def preprocess_remove_newLine(self, fasta):
        seq=""
        id_=""
        fasta=fasta.replace("_new","")
        g=open(fasta.split(".")[0]+"_new."+fasta.split(".")[1],"w")
        f=open(fasta,"r")
        for line in f:
            l=line.replace("\n","")
            if(l.startswith(">")):
                if(seq!=""):
                    g.write("%s\n%s\n" %(id_, seq) )
                id_=l.replace('\t','_').replace('.','_')
                seq=""
            else:
                seq+=l.replace('X','').replace('Z','')
        
        if(seq!=""):
            g.write("%s\n%s\n" %(id_, seq) )
        
        g.close()
        f.close()
    
    def _calculate_features(self, seq, mode):
        aac={}
        
        n=len(seq)
        
        l=1
        lags=list(range(1, l+1))
        a=[]
        for l in lags:
            aac["l"+str(l)]=[]
            
            for i in range(len(self.descriptors['M'])):
                init=0
                for s in seq:
                    init+=self.descriptors[s][i]
                aac["l"+str(l)].append(init)
        return aac                
                        
        
    def build_dataset_matrix_variance(self, mode, ide, method):
        fasta_pos=ide+"/"+method+"/dataset_pos.fasta"
        fasta_neg=ide+"/"+method+"/dataset_neg.fasta"
        
        classes={"pos": 1, "neg": 0}
        
        g=open(ide+"/"+method+"/"+mode+"_dataset.tsv", "w")
        feas=[]
        for i in range(1, 26):
            feas.append("feature"+str(i))
            
        g.write("protein\tclass\tlag\tfeatures\n")
        
        for cl in classes.keys():
            fasta=eval("fasta_"+cl)
            if(fasta!=""):
                class_=classes[cl]
                
                self.preprocess_remove_newLine(fasta)
                
                id_=""
                f=open(fasta.split(".")[0]+"_new."+fasta.split(".")[1], "r")
                for line in f:
                    l=line.replace("\n","")
                    if(not l.startswith(">")):
                        if(id_!=""):
                            data = eval("self._calculate_features(l, mode)")
                            for k in data.keys():
                                strar=[]
                                for e in data[k]:
                                    strar.append( "{:.5f}".format(e) )
                                    
                                features=','.join(strar)
                                g.write("%s\t%i\t%s\t%s\n" %(id_, class_, k, features) )
                    else:
                        try:
                            id_=l.replace(">","").split("|")[1].replace('\t','').replace('.','').replace('-','')
                        except:
                            try:
                                id_=l.replace(" ","").split("|")[0].replace('\t','').replace('.','').replace('-','')
                            except:
                                id_=""
                f.close()
            
        g.close()
        
class EvaluationModels:
    def _get_optimal_lag(self, fasta_pos, fasta_neg):
        seqs=set()
        f=open(fasta_pos, "r")
        for line in f:
            l=line.replace("\n","")
            if(not l.startswith(">")):
                seqs.add(l)
        f.close()
        
        f=open(fasta_neg, "r")
        for line in f:
            l=line.replace("\n","")
            if(not l.startswith(">")):
                seqs.add(l)
        f.close()
        
        n=len(seqs)
        i=30
        init=1
        while(init!=n):
            init=0
            for s in seqs:
                if(len(s)>=i):
                    init+=1
            i-=1        
        return i
    
    def dataset_training_evaluation_separated(self, ide, met):
        kernel = 1.0 * RBF(1.0)
        
        clfs={'svm-rbf': "svm.SVC(kernel='rbf')", 
              'sgd': 'SGDClassifier(loss="hinge", penalty="l2", max_iter=5)',
              'nearestCentroid': 'NearestCentroid()',
              'adaboost': 'AdaBoostClassifier()',
              #'gaussianProcess': 'GaussianProcessClassifier(kernel=kernel, random_state=0, n_jobs=-1)',
              'BernoulliNaiveBayes': 'BernoulliNB()',
              'DecisionTree': 'tree.DecisionTreeClassifier()',
              'GradientBoosting': "GradientBoostingClassifier( n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)",
              'NeuralNetwork': "MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)",
              'randomForest': 'RandomForestClassifier(max_depth=5)'
        }
        
        # Separated dataset of lags 
        fg=open(ide+"/"+met+"/"+"separated_model_results.tsv","w")
        fg.write("mode\tlag\tclassifier\tmean_accuracy\tmean_f1\tmean_precision\tmean_recall\tstdev_accuracy\tstdev_f1\tstdev_precision\tstdev_recall\n")
        modes =['auto','cross']
        maxi_lag=8
        if(met=='method2' or met=='method3'):
            modes =['auto']
            fasta_pos=ide+"/"+met+"/dataset_pos.fasta"
            fasta_neg=ide+"/"+met+"/dataset_neg.fasta"
            maxi_lag=self._get_optimal_lag(fasta_pos, fasta_neg)
            
        if(met=='method3'):
            maxi_lag=2
            
        for m in modes:
            for lag in range(1, maxi_lag+1):
                df=pd.read_csv(ide+"/"+met+"/"+""+m+"_dataset.tsv", sep="\t")
                x = df[ df['lag']=='l'+str(lag) ]
                if(len(x)>1):
                    y=x.iloc[:,1]
                    X=[]
                    for i in range(len(x)):
                        aux=[]
                        values=x.iloc[i,3].split(",")
                        for v in values:
                            aux.append(float(v))
                        X.append(aux)
                        
                    for classifier in clfs:
                        clf = eval(clfs[classifier])
                        clf.fit(X, y)
                        dump(clf, ide+"/"+met+'/models/'+m+'_l'+str(lag)+'_'+classifier+'_model_trained.joblib')
                        
                        f1=cross_val_score(clf, X, y, scoring='f1', cv=10)
                        precision=cross_val_score(clf, X, y, scoring='precision', cv=10)
                        recall=cross_val_score(clf, X, y, scoring='recall', cv=10) 
                        accuracy=cross_val_score(clf, X, y, scoring='accuracy', cv=10)
                        
                        fg.write("%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(m, 'l'+str(lag), classifier, st.mean(f1), st.mean(precision), st.mean(recall), st.mean(accuracy), st.stdev(f1), st.stdev(precision), st.stdev(recall), st.stdev(accuracy) ) )
                        
                        #for i in range(len(f1)):
                        #    fg.write(m+";l"+str(lag)+";"+classifier+";"+str(f1[i])+";"+str(precision[i])+";"+str(recall[i])+";"+str(accuracy[i])+"\n")
                        
                        ides=m+"_l"+str(lag)+"_"+classifier
        fg.close()
        
    def prepare_condensed_dataset(self, ides, met):
        modes =['auto','cross']
        if(met=='method2' or met=='method3'):
            modes =['auto']
        for m in modes:
            flag_stop=False
            f=open(ides+"/"+met+"/"+m+"_mapping_name_features.tsv", "w")
            f.close()
        
            index=1
            
            dfv=pd.read_csv(ides+"/"+met+"/"+m+"_dataset.tsv", sep="\t")
            """pieces=[]
            auxdfp=df[ (df['class']==1) ]
            auxdfn=df[ (df['class']==0) ]
            ns=len(auxdfp)
            if(len(auxdfp) > len(auxdfn)):
                ns=len(auxdfn)
            chosen=random.sample( list(auxdfp.index), ns)
            pieces.append(df.iloc[chosen, :])
            
            chosen=random.sample( list(auxdfn.index), ns)
            pieces.append(df.iloc[chosen, :])
                
            dfv=pd.concat( pieces )"""
            y=[]
            auxx={}
            aux=[]
            for i in range(len(dfv)):
                ide=dfv.iloc[i, 0]+'-'+str(dfv.iloc[i, 1])
                ide=ide.replace('\t','').replace(' ','').replace(':','')
                lag=dfv.iloc[i,2]
                
                if(not ide in auxx.keys()):
                    if(len(auxx.keys())>0):
                        flag_stop=True
                        
                    auxx[ide]=[]
                    y.append(dfv.iloc[i, 1])
                
                h=1
                aux=[]
                values=dfv.iloc[i,3].split(",")
                for v in values:
                    aux.append(str(v))
                    
                    if(not flag_stop):
                        with open(ides+"/"+met+"/"+m+"_mapping_name_features.tsv", "a") as f:
                            f.write("%s\t%s\n" %("f"+str(index), lag+"_f"+str(h)) )
                    h+=1
                    index+=1
                
                auxx[ide]+=aux
                
            f=open(ides+"/"+met+"/"+m+"_dataset_consolidated.tsv", "w")
            f.write("protein\tclass\tfeatures\n")
            for k in auxx.keys():
                info=k.split("-")
                f.write( ('\t'.join(info))+"\t"+(','.join(auxx[k]))+"\n")
            f.close()
            
    def execute_feature_selection(self, ide, met):
        modes =['auto','cross']
        if(met=='method2' or met=='method3'):
            modes =['auto']
        
        mapp={}
        for m in modes:
            mapp[m]={}
            f=open(ide+"/"+met+"/"+m+"_mapping_name_features.tsv", "r")
            for line in f:
                l=line.replace("\n","").split("\t")
                mapp[m][l[0]]=l[1]
            f.close()
        
        clfs={'randomForest': 'RandomForestClassifier(max_depth=5)',
              'GradientBoosting': "GradientBoostingClassifier( n_estimators=100, learning_rate=1.0, max_depth=5, random_state=0)"
        }
        
        X=[]
        Y=[]
        fg=open(ide+"/"+met+"/consolidated_featureSelection_model_results.tsv","w")
        fg.write("selected_features\tmode\tclassifier\tmean_accuracy\tmean_f1\tmean_precision\tmean_recall\tstdev_accuracy\tstdev_f1\tstdev_precision\tstdev_recall\n")
        
        for m in modes:
            df=pd.read_csv(ide+"/"+met+"/"+""+m+"_dataset_consolidated.tsv", sep="\t")
            pieces=[]
            auxdfp=df[ (df['class']==1) ]
            auxdfn=df[ (df['class']==0) ]
            ns=len(auxdfp)
            if(len(auxdfp) > len(auxdfn)):
                ns=len(auxdfn)
            chosen=random.sample( list(auxdfp.index), ns)
            pieces.append(df.iloc[chosen, :])
            
            chosen=random.sample( list(auxdfn.index), ns)
            pieces.append(df.iloc[chosen, :])
                
            dfv=pd.concat( pieces )
            Y=dfv.iloc[:,1]
            X=[]
            feas=[]
            for i in range(len(dfv)):
                aux=[]
                values=dfv.iloc[i,2].split(",")
                j=1
                for v in values:
                    aux.append(float(v))
                    if(i==0):
                        feas.append("f"+str(j))
                    j+=1
                
                X.append(aux)
                
            print(m, len(X[0]), len(feas))
            X=pd.DataFrame(X)
            #print(X)
            X.columns=feas
            X.to_csv(ide+"/"+met+"/"+"feature_selection_results/"+m+"_features_selection_dataset.tsv", sep='\t')
            
            for classifier in clfs.keys():
                print("-----", m, classifier)
                
                #forest = RandomForestClassifier(n_jobs=-1,  max_depth=5)
                forest=eval(clfs[classifier])
                forest.fit(X, Y)
                feat_selector = BorutaPy(forest, n_estimators='auto', random_state=1)
                feat_selector.fit(X.to_numpy(), Y)
                
                feature_ranks = list(zip(X.columns, feat_selector.ranking_, feat_selector.support_))
                sels=[]
                f=open(ide+"/"+met+"/"+classifier+"_"+m+"_features_selection_result.tsv","w")
                # iterate through and print out the results
                for feat in feature_ranks:
                    f.write('Feature: {:<25} Rank: {},  Keep: {} \n'.format(feat[0], feat[1], feat[2]))
                    if(feat[2]):
                        sels.append(mapp[m][feat[0]])
                f.close()
                    
                X_filtered = feat_selector.transform(X.to_numpy())
                if( len(X_filtered[0,:]) > 0):
                    X_test=X_filtered[:20000,:]
                    y_test=Y[:20000]
                    clf = AdaBoostClassifier()
                    clf.fit(X_filtered, Y)
                    dump(clf, ide+"/"+met+'/models/consolidated_'+m+'_'+classifier+'_model_trained.joblib')
                    
                    f1=cross_val_score(clf, X, Y, scoring='f1', cv=10)
                    precision=cross_val_score(clf, X, Y, scoring='precision', cv=10)
                    recall=cross_val_score(clf, X, Y, scoring='recall', cv=10) 
                    accuracy=cross_val_score(clf, X, Y, scoring='accuracy', cv=10)
                    
                    fg.write("%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %((','.join(sels)), m, classifier, st.mean(f1), st.mean(precision), st.mean(recall), st.mean(accuracy), st.stdev(f1), st.stdev(precision), st.stdev(recall), st.stdev(accuracy) ) )
                    
                    #predictions = clf.predict(X_test)
                    
                    #fg.write("%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\n" %((','.join(sels)), m, classifier, accuracy_score(y_test, predictions), f1_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test, predictions) ) )
                    
        fg.close()

class GramComparison:
    def preprocess_remove_newLine(self, fasta):
        seq=""
        id_=""
        fasta=fasta.replace("_new","")
        g=open(fasta.split(".")[0]+"_new."+fasta.split(".")[1],"w")
        f=open(fasta,"r")
        for line in f:
            l=line.replace("\n","")
            if(l.startswith(">")):
                if(seq!=""):
                    g.write("%s\n%s\n" %(id_, seq) )
                id_=l.replace('\t','_').replace('.','_').replace(" ",'')
                seq=""
            else:
                seq+=l.replace('X','').replace('Z','')

        if(seq!=""):
            g.write("%s\n%s\n" %(id_, seq) )

        g.close()
        f.close()
        
    def _init_folders_for_new_ds(self, identificador, m):
        os.system("mkdir "+identificador+"/"+m)
        os.system("mkdir "+identificador+"/"+m+"/models")
        os.system("cp "+identificador+"/dataset_* "+identificador+"/"+m+"/")
    
    def build_exps_datasets(self):
        # all pos vs all neg (gram+ with gram-):
        """if(not os.path.isdir('gram_comparison/all_gram')):
            os.system('mkdir gram_comparison/all_gram')
        
        for i in ['pos','neg']:
            f=open("gram_comparison/all_gram/dataset_"+i+".fasta","w")
            for j in ['gram+','gram-']:
                if(not os.path.isfile("gram_comparison/"+j+"_dataset_"+i+"_new.fasta")):
                    self.preprocess_remove_newLine("gram_comparison/"+j+"_dataset_"+i+".fasta")
                g=open("gram_comparison/"+j+"_dataset_"+i+"_new.fasta", "r")
                for line in g:
                    l=line.replace('\n','')
                    if(l.startswith('>')):
                        id_=l
                    else:
                        f.write('%s\n%s\n' %(id_, l))
                g.close()
            f.close()"""
        
        #datasets=[ 'gram_comparison/gram+','gram_comparison/gram-','gram_comparison/all_gram','gram_comparison/bcipep','gram_comparison/hla']
        #datasets=['gram_comparison/hla']
        datasets=[ 'gram_comparison/gram+','gram_comparison/gram-','gram_comparison/all_gram']
        methods=['method1', 'method2','method3']
        #methods=['method3']
        
        ev=EvaluationModels()
        
        met1=Implementation_vaxijen()
        met2=Implementation_vaxijenModified()
        met3=Implementation_new_158()
        
        for ds in datasets:
            for m in methods:
                if(not os.path.isdir(ds+"/"+m)):
                    self._init_folders_for_new_ds(ds, m)    
                    print("----------- Preparing dataset")
                    modes =['auto','cross']
                    instance=met1
                    if(m=='method2'):
                        modes = ['auto']
                        instance=met2
                    if(m=='method3'):
                        modes = ['auto']
                        instance=met3
                        
                    for mo in modes:
                        print("\t", ds, ' - ', m, ' - ', mo)
                        instance.build_dataset_matrix_variance(mo, ds, m)  
                    
                    print('---- evaluating separated', ds, m)
                    ev.dataset_training_evaluation_separated(ds, m)  
                    
                    print('---- preparing condensed', ds, m)
                    ev.prepare_condensed_dataset(ds, m)
                    
                    print('---- evaluating c', ds, m)
                    ev.execute_feature_selection(ds, m)
        
        # vaxign-ml gram- train with gram+ for test and vice-versa
        # rank each model, choose the best among the separated and the consolidated
        # use the two models of one mode (all_gram, gram-, gram+) to test with the others
        # align the sequences gram+ pos with gram- pos, gram+ neg with gram- neg

    def count_accuracy_vaxignml_mixedGram(self):
        f=open("gram_comparison/results_vaxignml.tsv","w")
        f.write("train\ttest\taccuracy\tf1\tprecision\trecall\n")
        for tr in ['Gram+', 'Gram-']:
            for te in ['neg','pos']:
                cls=0.0
                if(te=='pos'):
                    cls=1.0
                    
                y=[]
                preds=[]
                i=0
                g=open("Vaxign-ML-docker/out_train"+tr+"_testall_"+te+"/dataset_"+te+".result.tsv")
                for line in g:
                    if(i>0):
                        l=float(line.split("\t")[1])
                        preds.append(l)
                        y.append(cls)
                    i+=1
                g.close()
                f.write( "%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\n" %(tr, te, accuracy_score(y,preds), f1_score(y,preds), precision_score(y,preds), recall_score(y,preds)) )
        f.close()

class SequenceComposition:
    
    def count_kmers(self):
        """
        Test the occurrences number of kmers of 4 to 8 length for negative and positive, as well the gram distributions
        """
        datasets=[ 'gram_comparison/gram+','gram_comparison/gram-','gram_comparison/all_gram']
        for ds in datasets:
            for clas in ['pos','neg']:
                acc={}
                seqs={}
                c=0
                f=open(ds+"/dataset_"+clas+".fasta", "r")
                for line in f:
                    if(not line.startswith(">")):
                        aux=[]
                        l=line.replace('\n','').replace('X','').replace('Z','')
                        for k in range(2, 6):
                            for i in range(0,len(l)-k):
                                kmer=l[i:i+k]
                                if(not kmer in acc.keys()):
                                    acc[kmer]=0
                                acc[kmer]+=1
                                
                                if(not kmer in aux):
                                    aux.append(kmer)
                        for a in aux:
                            if(not a in seqs.keys()):
                                seqs[a]=0
                            seqs[a]+=1
                        c+=1
                f.close()
                
                f=open(ds+"/kmer_"+clas+".tsv", "w")
                f.write("sequence\toccurrences\tk-length\toccurrences\tcoverage\ttotal_sequences\n")
                for k in acc.keys():
                    f.write("%s\t%s\t%s\t%i\t%.5f\t%i\n" %(k, acc[k], len(k), seqs[k], seqs[k]/c, c) )
                f.close()
 
    def count_aa_distribution(self):
        """
        Test the occurrences number of aminoacid distributions
        """
        datasets=[ 'gram_comparison/gram+','gram_comparison/gram-','gram_comparison/all_gram']
        for ds in datasets:
            for clas in ['pos','neg']:
                acc={}
                f=open(ds+"/dataset_"+clas+".fasta", "r")
                for line in f:
                    if(not line.startswith(">")):
                        l=line.replace('\n','').replace('X','').replace('Z','')
                        for aa in l:
                            if(not aa in acc.keys()):
                                acc[aa]=0
                            acc[aa]+=1
                f.close()
                
                f=open(ds+"/aa_"+clas+".tsv", "w")
                f.write("aa\toccurrences\n")
                for k in acc.keys():
                    f.write("%s\t%s\n" %(k, acc[k]) )
                f.close()

    def evaluate_diff_inter_kmers(self):
        folder="gram_comparison"    
        datasets=[ 'gram+','gram-','all_gram']
        aux={}
        for ds in datasets:
            for clas in ['pos','neg']:
                aux[ds+"_"+clas]=set()
                i=0
                f=open(folder+"/"+ds+"/kmer_"+clas+".tsv", "r")
                for line in f:
                    if(i>0):
                        l=line.replace("\n","").split("\t")
                        if(float(l[1])>10):
                            aux[ds+"_"+clas].add(l[0])
                    i+=1
                f.close()
        
        f=open(folder+"/sets_comparison_kmers.tsv","w")  
        f.write("set1\tset2\tintersection\tdifference\tnumber_inter\tnumber_diff\n")   
        dupp=set()
        for k in aux.keys():
            for v in aux.keys():
                if(not k+"-"+v in dupp and k!=v):
                    dupp.add(k+"-"+v)
                    dupp.add(v+"-"+k)
                    f.write("%s\t%s\t%s\t%s\t%i\t%i\n" %(k, v, ','.join(list(aux[k].intersection(aux[v]))), ','.join(list(aux[k].difference(aux[v]))), len(aux[k].intersection(aux[v])), len(aux[k].difference(aux[v])) ) )
        f.close()            

    def select_models(self):
        folder="gram_comparison"    
        datasets=[ 'gram+','gram-','all_gram','bcipep','hla']
        datasets=[ 'gram+','gram-','all_gram']
        methods=['method1','method2','method3']
        aux={}
        for ds in datasets:
            if(not os.path.isdir(folder+"/"+ds+"/best_models")):
                os.system("mkdir "+folder+"/"+ds+"/best_models")
                
            for m in methods:
                df=pd.read_csv(folder+"/"+ds+"/"+m+'/separated_model_results.tsv', sep='\t')
                sdf=df.sort_values(by=['mean_f1'])
                for i in range(3):
                    mode = sdf.iloc[i,0]
                    lag = sdf.iloc[i,1]
                    os.system('cp '+folder+"/"+ds+"/"+m+'/models/'+mode+"_"+lag.replace('l','')+"_model_trained.joblib "+folder+"/"+ds+'/best_models/'+str(i)+"-"+m+'-'+mode+"-"+lag+'-separated_model.joblib')
                
                """
                df=pd.read_csv(folder+"/"+ds+"/"+m+'/consolidated_featureSelection_model_results.tsv', sep='\t')
                sdf=df.sort_values(by=['mean_f1'])
                mode = sdf.iloc[0,1]
                cls = sdf.iloc[0,2]
                os.system('cp '+folder+"/"+ds+"/"+m+'/models/consolidated_'+mode+"_"+cls+"_model_trained.joblib "+folder+"/"+ds+'/best_models/'+m+'_consolidated_model.joblib')
                """
                
    def test_models(self):
        folder="gram_comparison"    
        #datasets=[ 'gram+','gram-','all_gram','bcipep','hla']
        #datasets=[ 'gram+','gram-','all_gram']
        datasets=[ 'gram+','gram-','all_gram']
        methods=['method1','method2']
        params={ 'method1': { 'cross': 20, 'auto': 5 }, 'method2': { 'auto': 6 }, 'method3': { 'auto': 178 } }
        passed=set()
        aux={}
        res=open(folder+"/comparison_models_gram.tsv","w")
        res.write("ranking\tsource\ttarget\tmethod\tmode\tlag\taccuracy\tf1\tprecision\trecall\tauc\n")
        for ds in datasets:
            if(not os.path.isdir(folder+"/"+ds+"/best_models/results")):
                os.system("mkdir "+folder+"/"+ds+"/best_models/results")
            for d in datasets:
                if(ds!=d):
                    if(not os.path.isdir(folder+"/"+ds+"/best_models/results/"+d)):
                        os.system("mkdir "+folder+"/"+ds+"/best_models/results/"+d)
                        
                    for m in methods:
                        for mo in os.listdir(folder+"/"+ds+"/best_models"):
                            if(mo.find(m)!=-1):
                                name = mo.split(".")[0].replace('-separated_model','')
                                
                                index='top-'+str(int(name.split('-')[0])+1)
                                mode=name.split('-')[2]
                                lag=name.split('-')[3]
                                df=pd.read_csv(folder+"/"+ds+"/"+m+"/"+mode+'_dataset.tsv',sep='\t')
                                print(ds, d, m, mode)
                                filt=df[ df['lag']==lag ]
                                if(len(filt)>0):
                                    p=0
                                    n=0
                                    y=[]
                                    ys=filt['class'] # original class
                                    for v in ys:
                                        if(v>0.9):
                                            y.append(1)
                                            p+=1
                                        else:
                                            y.append(0)
                                            n+=1
                                    
                                    cols=[] # making columns for the dataframe of the filtered lag dataset
                                    ncols=params[m][mode]
                                    for c in range(1,ncols+1):
                                        cols.append('f'+str(c))
                                        
                                    feas=filt['features'] # getting float values of the features
                                    X=[]
                                    for f in feas:
                                        values=f.split(",")
                                        aux=[]
                                        for v in values:
                                            aux.append(float(v))
                                        X.append(aux)
                                    tempdf=pd.DataFrame(X)
                                    tempdf.columns=cols
                                    
                                    model=load(folder+"/"+ds+"/best_models/"+mo) # loading model
                                    preds = model.predict(tempdf) # predicting
                                    preds_ = model.predict_proba(tempdf) # predicting
                                    #print(Counter(y), p, n)
                                    tempdf['y']=y
                                    tempdf['preds']=preds
                                    
                                    #fpr, tpr, thresholds = roc_curve(y, preds_[:,1], pos_label=2)
                                    mauc=roc_auc_score(y, preds_[:,1])
                                    
                                    tempdf.to_csv(folder+"/"+ds+"/best_models/results/"+d+"/"+name+'_dataset.tsv',sep='\t')
                                    
                                    res.write("%s\t%s\t%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(index, ds, d, m, mode, lag, accuracy_score(y,preds), f1_score(y,preds), precision_score(y,preds), recall_score(y,preds), mauc ) )
        res.close()
    
    def count_kmer(self, seq, kmer):
        count=0
        n=len(kmer)
        for i in range(0,len(seq)-n):
            current_kmer=seq[i:i+n]
            if(current_kmer==kmer):
                count+=1
        return count
    
    def test_clustering_kmers(self):
        folder="gram_comparison"    
        datasets=[ 'gram+','gram-','all_gram']
        classes=['neg','pos']
        
        dat={}
        for d in datasets:
            for c in classes:
                # select kmers with more coverage 
                df=pd.read_csv(folder+'/'+d+"/kmer_"+c+".tsv", sep='\t')
                filt=df[ df['coverage']>=0.8 ]
                dat[d+"|"+c]=set(filt['sequence'])
        
        res=open(folder+"/frequency_comparison_clustering_gram.tsv","w")
        res.write("set1\tset2\taccuracy\tf1\tprecision\trecall\tauc\n")        
        res1=open(folder+"/binary_comparison_clustering_gram.tsv","w")
        res1.write("set1\tset2\taccuracy\tf1\tprecision\trecall\tauc\n")        
        passed=set()
        for k in dat.keys():
            for v in dat.keys():
                if(k!=v and not k+"_"+v in passed):
                    passed.add(k+"_"+v)
                    passed.add(v+"_"+k)
                    
                    features=dat[k].union(dat[v])
                    
                    # frequency
                    X=[]
                    y=[]
                    j=0
                    for comp in [k, v]:
                        ds = comp.split('|')[0]
                        clas = comp.split('|')[1]
                        f=open(folder+"/"+ds+"/dataset_"+clas+".fasta", "r")
                        for line in f:
                            if(not line.startswith(">")):
                                aux=[]
                                l=line.replace('\n','').replace('X','').replace('Z','')
                                for kmer in features:
                                    aux.append(self.count_kmer(l, kmer))
                                X.append(aux)
                                y.append(j)
                        f.close()
                        j+=1
                       
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
                    preds = kmeans.labels_
                    res.write("%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\n" %(k, v, accuracy_score(y,preds), f1_score(y,preds), precision_score(y,preds), recall_score(y,preds) ) )
                    
                    # binary
                    X=[]
                    y=[]
                    j=0
                    for comp in [k, v]:
                        ds = comp.split('|')[0]
                        clas = comp.split('|')[1]
                        f=open(folder+"/"+ds+"/dataset_"+clas+".fasta", "r")
                        for line in f:
                            if(not line.startswith(">")):
                                aux=[]
                                l=line.replace('\n','').replace('X','').replace('Z','')
                                for kmer in features:
                                    if(l.count(kmer)>0):
                                        aux.append(1)
                                    else:
                                        aux.append(0)
                                X.append(aux)
                                y.append(j)
                        f.close()
                        j+=1
                    
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
                    preds = kmeans.labels_
                    res1.write("%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\n" %(k, v, accuracy_score(y,preds), f1_score(y,preds), precision_score(y,preds), recall_score(y,preds) ) )
        res.close()
        res1.close()
    
    def _count_epitopes_in_sec(self, epitopes_bac, seq, counts, promiscuous):
        for cl in ['pos', 'neg']:
            for s in epitopes_bac[cl]:
                if(seq.find(s)!=-1):
                    counts[cl]+=1
                    promiscuous[s]+=1
        return counts, promiscuous
        
    def check_presence_epitopes_bacteria_proteins(self):
        promiscuous={}
        epitopes_bac={ 'pos': set(), 'neg': set() }
        f=open("bacteria","r")
        for line in f:
            l=line.replace('\n','')
            if(line.startswith('Sequence')):
                seq=l.split(' ')[-1]
            if(line.startswith('Immunogenicity')):
                promiscuous[seq]=0
                
                clas=l.split(' ')[-1]
                if(clas=='No'):
                    epitopes_bac['neg'].add(seq)
                else:
                    epitopes_bac['pos'].add(seq)
        f.close()
        print('Neg bac epitopes:', len(epitopes_bac['neg']) )
        print('Pos bac epitopes:', len(epitopes_bac['pos']) )
        
        columns='\t'.join(promiscuous.keys())
        h=open('report_epitopes_counts.tsv','w')
        h.write('dataset\tclass\t'+columns+'\n')
        
        g=open('report_epitopes_ambiguous_classes.tsv','w')
        g.write('dataset\tclass\tcount_epitopes_pos\tcount_epitopes_neg\n')
        dss=['gram+_dataset', 'gram-_dataset']
        for d in dss:
            for cl in ['pos', 'neg']:
                aux=promiscuous
                counts={'pos': 0, 'neg': 0}
                f=open(d+'/dataset_'+cl+'.fasta','r')
                for line in f:
                    l=line.replace('\n','')
                    if(not l.startswith('>')):
                        counts, aux = self._count_epitopes_in_sec( epitopes_bac, l, counts, aux)
                f.close()
                
                temp=[]
                for v in promiscuous.values():
                    temp.append(str(v))
                h.write(d+'\t'+cl+'\t'+('\t'.join(temp))+'\n')    
                    
                g.write('%s\t%s\t%i\t%i\n' %(d, cl, counts['pos'], counts['neg']) )
        g.close()
        h.close()
      
from collections import Counter                          
a=GramComparison()
#a.build_exps_datasets() # make the prediction and save models

#a.count_accuracy_vaxignml_mixedGram()

b=SequenceComposition()
#b.count_kmers()
#b.count_aa_distribution()
#b.evaluate_diff_inter_kmers()

#b.select_models() # select the best models based on mean f1
b.test_models() # Use the best trained models across the other datasets

#b.check_presence_epitopes_bacteria_proteins()#

#b.test_clustering_kmers()

