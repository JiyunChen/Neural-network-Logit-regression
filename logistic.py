# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:10:16 2018

@author: chenj
"""
import math
import sys
import numpy as np
import random
from collections import OrderedDict
from scipy.io import arff
import pandas as pd
#import copy
def readArff(filename):
    trainset,meta = arff.loadarff(filename)
    train=[]
    for i in range(len(trainset)):
        temp=trainset[i];s=[]
        for j in range(len(temp)):
            ty=type(temp[-1])
            if type(temp[j])==ty:
                #print(1)
                s.append(temp[j].decode())
            else:
                s.append(temp[j])
        train.append(s)     
    feature=OrderedDict()
    for i in meta.names():
        if meta[i][1]==None:
            feature[i]=[]
        else:
            feature[i]=list(meta[i][1])
    return[train,meta.names(),feature]

def get_y(train,feature):
    index=feature['class'][0]
    for i in train:
        if i[-1]==index:
            i[-1]=0
        else:
            i[-1]=1
    return train

def nominal(data,name,feature):
    dat=pd.DataFrame(data,columns=name);l=[];
    for i in name[:-1]:
        if len(feature[i])!=0:
            l.append(i)
    #print('class')
    if len(l)==0:
        #print('yes')
        return np.array(dat).tolist()
    else:
        #print(l)
        c=dat['class'];del dat['class']
        for index in l:
            #print(1)
            x=dat[index];save=[]
            #print(x)
            del dat[index];#print(feature[index])
            for i in feature[index]:
                #print(i)
                temp=[]
                for j in x:
                    #print(j)
                    if j == i:
                        temp.append(1)
                    else:
                        temp.append(0)
                save.append(temp)
            #print(len(save[1]))
            use=pd.DataFrame(save).T
        #print(use.shape)
            dat = pd.concat([dat, use], axis=1)
        #print(dat.shape)
        cl=pd.DataFrame(c)
        #print(cl.shape)
        dat=pd.concat([dat,cl],axis=1)
        return np.array(dat).tolist()

def standardlize(train):
    t=np.array(train)            
    mean=[];sd=[];
    for i in range(len(t[0])-1):
        mean.append(np.mean(t[:,i]));
        if np.std(t[:,i])==0:
            sd.append(1)
        else:
            sd.append(np.std(t[:,i]))
    return [mean,sd]

def stand_data(dat,train):
    mean=standardlize(train)[0]
    sd=standardlize(train)[1]
    new=[]
    for i in dat:
        x=np.array(i[:-1]);y=i[-1]
        i_new=(x-mean)/sd
        i_new=list(i_new)
        i_new.append(y)
        new.append(i_new)
    return new

def random_weight(n):
    w=[]
    for i in range(n):
        w.append(random.uniform(-0.01,0.01))
    return(np.array(w))

def sigmoid(x):
    y=1/(1+math.e**(-x))
    return(y)

def Logit(train,l,e):
    w=random_weight(len(train[0]))
    epoch=0;c_e=[];num=[];num_f=[]
    while epoch < e:
        #print(e)
        n=0
        order=random.sample(range(len(train)), len(train)); new_train=[]
        for i in order:
            new_train.append(train[i])
        entropy=0;
        for i in new_train:
            x=np.array([1]+i[:-1])
            #print(w)
            o_d=sigmoid(sum(x*w))
            if o_d > 0.5:
                p_d=1
            else:
                p_d=0
            #print(x*w)
            y_d=i[-1]
            if y_d==p_d:
                n=n+1
            w_tri=(l*(y_d-o_d))*x
            w=w+w_tri
            #print(y_d,o_d)
            #print(w)
            entropy_d=(-y_d)*math.log(o_d,math.e)-(1-y_d)*math.log((1-o_d),math.e)
            entropy=entropy+entropy_d
        c_e.append(entropy);num.append(n);num_f.append(len(train)-n)
        epoch=epoch+1
    return [w,c_e,num,num_f]

def predict_log(w,test):
    ans=[];class_l=[];n=0;actual=[] 
    for i in test:
        x=np.array([1]+i[:-1]);y_d=i[-1]
        o_d=sigmoid(sum(x*w))
        ans.append(o_d);actual.append(y_d)
        if o_d > 0.5:
            p_d=1
            class_l.append(1)
        else:
            p_d=0
            class_l.append(0)
        if p_d==y_d:
            n=n+1
    return [ans,class_l,actual,n,len(test)-n]

def F1(pre,actual):
    dat=pd.DataFrame()
    #dat.insert(0,"cond_pos",ans)
    dat.insert(0,'predict',pre)
    dat.insert(1,'actual',actual)
    temp=dat.loc[dat['actual']==1]
    recall=temp.loc[temp['predict']==1].shape[0]/temp.shape[0]
    precision=temp.loc[temp['predict']==1].shape[0]/dat.loc[dat['predict']==1].shape[0]
    #print(precision,recall)
    F=(2*(precision*recall))/(precision + recall)
    #print(F)
    return F
    
def print_log(entropy,n,n_f,predict_ans,pre,actual,n_2,n_2_f,F1):
    for i in range(len(entropy)):
        print(i+1,'\t',format(entropy[i],'0.9f'),'\t',n[i],'\t',n_f[i])
    for i in range(len(predict_ans)):
        print(format(predict_ans[i],'0.9f'),'\t',pre[i],int(actual[i]))
    print(n_2,'\t',n_2_f)
    print(format(F1,'0.12f'))

def main():
    d=readArff(sys.argv[3])
    test=readArff(sys.argv[4])
    new_train=get_y(d[0],d[2])
    new_train=nominal(new_train,d[1],d[2])
    new_stand_train=stand_data(new_train,new_train)
    l=float(sys.argv[1]);e=int(sys.argv[2])
    res=Logit(new_stand_train,l,e)
    new_test=get_y(test[0],test[2])
    new_test=nominal(new_test,test[1],test[2])
    new_stand_test=stand_data(new_test,new_train) 
    pre=predict_log(res[0],new_stand_test)
    F1_score=F1(pre[1],pre[2])
    print_log(res[1],res[2],res[3],pre[0],pre[1],pre[2],pre[3],pre[4],F1_score)
    #return F1_score

#a=main(0.1,10,"credit_train.arff","credit_test.arff")
#a=main(0.1,10,"diabetes_train.arff","diabetes_test.arff")
#a=main(0.01,3,'magic_train.arff','magic_test.arff')

if __name__ == '__main__':
    main()      