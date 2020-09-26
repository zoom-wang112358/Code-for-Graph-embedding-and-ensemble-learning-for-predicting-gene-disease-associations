# coding=UTF-8
import gc
import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import random
import os
from sklearn import preprocessing
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import deepwalk
from sklearn.decomposition import PCA
import argparse
import networkx as nx
# import node2vec
from openne import graph, node2vec, gf, lap, hope, sdne
import math


#droprate=0.4
def get_embedding(vectors: dict):
    matrix = np.zeros((
        12118,
        len(list(vectors.values())[0])
    ))
    print("axis 0:")
    print(len(vectors))
    print("axis 1:")
    print(len(list(vectors.values())[0]))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix


def get_embedding_lap(vectors: dict):
    matrix = np.zeros((
        12118,
        128
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix

def processEmb(oldfile,newfile):
    f = open(oldfile)
    next(f)
    for line in f:
        f1 = open(newfile,'a+')
        f1.write(line)
    f1.close()
    f.close()

def clearEmb(newfile):
    f = open(newfile,'w')
    f.truncate()

def Net2edgelist(gene_disease_matrix_net):
    none_zero_position = np.where(np.triu(gene_disease_matrix_net) != 0)
    none_zero_row_index = np.mat(none_zero_position[0],dtype=int).T
    none_zero_col_index = np.mat(none_zero_position[1],dtype=int).T
    none_zero_position = np.hstack((none_zero_row_index,none_zero_col_index))
    none_zero_position = np.array(none_zero_position)
    name = 'gene_disease.txt'
    np.savetxt(name, none_zero_position,fmt="%d",delimiter=' ')

#获得gene_disease_emb
def Get_embedding_Matrix_sdne(gene_disease_matrix_net):
    Net2edgelist(gene_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("gene_disease.txt")
    print(graph1)
    _sdne=Get_sdne(graph1)
    return _sdne
   
def Get_embedding_Matrix_gf(gene_disease_matrix_net):
    Net2edgelist(gene_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("gene_disease.txt")
    print(graph1)
    _gf=Get_gf(graph1)
    return _gf
    
def Get_embedding_Matrix_n2v(gene_disease_matrix_net):
    Net2edgelist(gene_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("gene_disease.txt")
    print(graph1)
    _n2v=Get_n2v(graph1)
    return _n2v

def Get_embedding_Matrix_dw(gene_disease_matrix_net):
    Net2edgelist(gene_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("gene_disease.txt")
    print(graph1)
    _dw=Get_dw(graph1)
    return _dw
    
def Get_embedding_Matrix_lap(gene_disease_matrix_net):
    Net2edgelist(gene_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("gene_disease.txt")
    print(graph1)
    _lap=Get_lap(graph1)
    return _lap
    
def Get_embedding_Matrix_hope(gene_disease_matrix_net):
    Net2edgelist(gene_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("gene_disease.txt")
    print(graph1)
    _hope=Get_hope(graph1)
    return _hope
    
def Get_sdne(graph1):
    model = sdne.SDNE(graph1, [1000, 128])
    return get_embedding(model.vectors)

def Get_n2v(graph1):
    model = node2vec.Node2vec(graph=graph1, path_length=80, num_paths=10, dim=128)
    n2v_vectors = get_embedding(model.vectors)
    return n2v_vectors

def Get_dw(graph1):
    model = node2vec.Node2vec(graph=graph1, path_length=80, num_paths=10, dim=128, dw=True)
    n2v_vectors = get_embedding(model.vectors)
    return n2v_vectors

def Get_gf(graph1):
    model = gf.GraphFactorization(graph1)
    return get_embedding(model.vectors)

def Get_lap(graph1):
    model = lap.LaplacianEigenmaps(graph1)
    return get_embedding_lap(model.vectors)

def Get_hope(graph1):
    model = hope.HOPE(graph=graph1, d=128)
    return get_embedding(model.vectors)


def get_gaussian_feature(A_B_matrix):
    row=A_B_matrix.shape[0]
    column=A_B_matrix.shape[1]
    A_matrix=np.zeros((row,row))
    for i in range(0,row):
        for j in range(0,row):
            A_matrix[i,j]=math.exp(-np.linalg.norm(np.array(A_B_matrix[i,:]-A_B_matrix[j,:]))**2)
    B_matrix=np.zeros((column,column))
    for i in range(0,column):
        for j in range(0,column):
            B_matrix[i,j]=math.exp(-np.linalg.norm(np.array(A_B_matrix[:,i]-A_B_matrix[:,j]))**2)
    A_matrix=np.matrix(A_matrix)
    B_matrix=np.matrix(B_matrix)
    return A_matrix,B_matrix

def make_prediction1(train_feature_matrix, train_label_vector,test_feature_matrix):
    clf = RandomForestClassifier(random_state=1, n_estimators=200, oob_score=True, n_jobs=-1)
    clf.fit(train_feature_matrix, train_label_vector)
    predict_y_proba = np.array(clf.predict_proba(test_feature_matrix)[:, 1])
    return predict_y_proba

def make_prediction2(train_feature_matrix, train_label_vector,test_feature_matrix):
    clf = MLPClassifier(solver='adam',activation = 'relu',max_iter = 100,alpha = 1e-5,hidden_layer_sizes = (128,64),verbose = False,early_stopping=True)
    clf.fit(train_feature_matrix, train_label_vector)
    predict_y_proba = np.array(clf.predict_proba(test_feature_matrix)[:,1])
    return predict_y_proba
    
def constructNet(gene_dis_matrix,dis_chemical_matrix,gene_chemical_matrix,gene_gene_matrix):
    disease_matrix = np.matrix(np.zeros((dis_chemical_matrix.shape[0], dis_chemical_matrix.shape[0]), dtype=np.int8))
    chemical_matrix = np.matrix(np.zeros((dis_chemical_matrix.shape[1], dis_chemical_matrix.shape[1]),dtype=np.int8))
    mat1 = np.hstack((gene_gene_matrix,gene_chemical_matrix,gene_dis_matrix))
    mat2 = np.hstack((gene_chemical_matrix.T,chemical_matrix,dis_chemical_matrix.T))
    mat3 = np.hstack((gene_dis_matrix.T,dis_chemical_matrix,disease_matrix))
    return np.vstack((mat1,mat2,mat3))


    

def calculate_metric_score(real_labels,predict_score):
    # evaluate the prediction performance
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
    aupr_score = auc(recall, precision)
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
       if (precision[k] + recall[k]) > 0:
           all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
       else:
           all_F_measure[k] = 0
    print("all_F_measure: ")
    print(all_F_measure)
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
    auc_score = auc(fpr, tpr)

    f = f1_score(real_labels, predict_score)
    print("F_measure:"+str(all_F_measure[max_index]))
    print("f-score:"+str(f))
    accuracy = accuracy_score(real_labels, predict_score)
    precision = precision_score(real_labels, predict_score)
    recall = recall_score(real_labels, predict_score)
    print('results for feature:' + 'weighted_scoring')
    print(    '************************AUC score:%.3f, AUPR score:%.3f, precision score:%.3f, recall score:%.3f, f score:%.3f,accuracy:%.3f************************' % (
        auc_score, aupr_score, precision, recall, f, accuracy))
    results = [auc_score, aupr_score, precision, recall,  f, accuracy]

    return results    
def ensemble_scoring(test_label_vector, predict_y,predict_y_proba):  # 计算3种集成方法的scores
    AUPR = average_precision_score(test_label_vector, predict_y_proba)
    AUC = roc_auc_score(test_label_vector, predict_y_proba)
    MCC = matthews_corrcoef(test_label_vector, predict_y)
    ACC = accuracy_score(test_label_vector, predict_y, normalize=True)
    F1 = f1_score(test_label_vector, predict_y, average='binary')
    REC = recall_score(test_label_vector, predict_y, average='binary')
    PRE = precision_score(test_label_vector, predict_y, average='binary')
    metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))

    return metric


def cross_validation_experiment(gene_dis_matrix,dis_chemical_matrix,gene_chemical_matrix,gene_gene_matrix,seed,ratio = 1):
    none_zero_position = np.where(gene_dis_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    zero_position = np.where(gene_dis_matrix == 0)
    zero_row_index = zero_position[0]
    zero_col_index = zero_position[1]
    random.seed(seed)
    zero_random_index = random.sample(range(len(zero_row_index)), ratio * len(none_zero_row_index))
    zero_row_index = zero_row_index[zero_random_index]
    zero_col_index = zero_col_index[zero_random_index]

    row_index = np.append(none_zero_row_index, zero_row_index)
    col_index = np.append(none_zero_col_index, zero_col_index)

    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    metric = np.zeros((1,7), float)
    print("seed=%d, evaluating gene-disease...." % (seed))
    k_count=0
    metric = np.zeros((1, 7))
    metric_csv = []
    times=1
    for train, test in kf.split(row_index):

        train_gene_dis_matrix = np.copy(gene_dis_matrix)

        test_row = row_index[test]
        test_col = col_index[test]
        train_row = row_index[train]
        train_col = col_index[train]

        train_gene_dis_matrix[test_row, test_col] = 0
        gene_disease_matrix_net = constructNet(train_gene_dis_matrix, dis_chemical_matrix, gene_chemical_matrix,
                                               gene_gene_matrix)
        print(gene_disease_matrix_net.shape)

        gene_dis_emb_sdne=Get_embedding_Matrix_sdne(np.mat(gene_disease_matrix_net))
        gene_dis_emb_n2v=Get_embedding_Matrix_n2v(np.mat(gene_disease_matrix_net))
        gene_dis_emb_dw=Get_embedding_Matrix_dw(np.mat(gene_disease_matrix_net))
        gene_dis_emb_lap=Get_embedding_Matrix_lap(np.mat(gene_disease_matrix_net))
        gene_dis_emb_hope=Get_embedding_Matrix_hope(np.mat(gene_disease_matrix_net))
        gene_dis_emb_gf=Get_embedding_Matrix_gf(np.mat(gene_disease_matrix_net))
        gene_len = gene_dis_matrix.shape[0]
        chem_len = gene_chemical_matrix.shape[1]
        
        times=times+1
        
        
        
        #gene_dis_emb_gf
        gene_emb_matrix_sdne = np.array(gene_dis_emb_sdne[0:gene_len, 0:])
        dis_emb_matrix_sdne = np.array(gene_dis_emb_sdne[(gene_len + chem_len)::, 0:])
        
        gene_emb_matrix_gf = np.array(gene_dis_emb_gf[0:gene_len, 0:])
        dis_emb_matrix_gf = np.array(gene_dis_emb_gf[(gene_len + chem_len)::, 0:])
        
        gene_emb_matrix_n2v = np.array(gene_dis_emb_n2v[0:gene_len, 0:])
        dis_emb_matrix_n2v = np.array(gene_dis_emb_n2v[(gene_len + chem_len)::, 0:])
        
        gene_emb_matrix_dw = np.array(gene_dis_emb_dw[0:gene_len, 0:])
        dis_emb_matrix_dw = np.array(gene_dis_emb_dw[(gene_len + chem_len)::, 0:])
        
        gene_emb_matrix_lap = np.array(gene_dis_emb_lap[0:gene_len, 0:])
        dis_emb_matrix_lap = np.array(gene_dis_emb_lap[(gene_len + chem_len)::, 0:])
        
        gene_emb_matrix_hope = np.array(gene_dis_emb_hope[0:gene_len, 0:])
        dis_emb_matrix_hope = np.array(gene_dis_emb_hope[(gene_len + chem_len)::, 0:])

        train_feature_matrix_sdne = []
        train_feature_matrix_gf = []
        train_feature_matrix_dw = []
        train_feature_matrix_n2v = []
        train_feature_matrix_lap = []
        train_feature_matrix_hope = []
        train_label_vector = []


        
        for num in range(len(train_row)):
            feature_sdne_gene=np.array(gene_emb_matrix_sdne[train_row[num], :])
            feature_sdne_dis=np.array(dis_emb_matrix_sdne[train_col[num], :])
            feature_gf_gene=np.array(gene_emb_matrix_gf[train_row[num], :])
            feature_gf_dis=np.array(dis_emb_matrix_gf[train_col[num], :])
            feature_dw_gene=np.array(gene_emb_matrix_dw[train_row[num], :])
            feature_dw_dis=np.array(dis_emb_matrix_dw[train_col[num], :])
            feature_n2v_gene=np.array(gene_emb_matrix_n2v[train_row[num], :])
            feature_n2v_dis=np.array(dis_emb_matrix_n2v[train_col[num], :])
            feature_lap_gene=np.array(gene_emb_matrix_lap[train_row[num], :])
            feature_lap_dis=np.array(dis_emb_matrix_lap[train_col[num], :])
            feature_hope_gene=np.array(gene_emb_matrix_hope[train_row[num], :])
            feature_hope_dis=np.array(dis_emb_matrix_hope[train_col[num], :])
            
            feature_vector_sdne =np.append(feature_sdne_gene, feature_sdne_dis)
            feature_vector_gf =np.append(feature_gf_gene, feature_gf_dis)
            feature_vector_dw =np.append(feature_dw_gene, feature_dw_dis)
            feature_vector_n2v =np.append(feature_n2v_gene, feature_n2v_dis)
            feature_vector_lap =np.append(feature_lap_gene, feature_lap_dis)
            feature_vector_hope =np.append(feature_hope_gene, feature_hope_dis)
            train_feature_matrix_sdne.append(feature_vector_sdne)
            train_feature_matrix_gf.append(feature_vector_gf)
            train_feature_matrix_dw.append(feature_vector_dw)
            train_feature_matrix_n2v.append(feature_vector_n2v)
            train_feature_matrix_lap.append(feature_vector_lap)
            train_feature_matrix_hope.append(feature_vector_hope)

            train_label_vector.append(gene_dis_matrix[train_row[num], train_col[num]])

        test_feature_matrix_sdne = []
        test_feature_matrix_gf = []
        test_feature_matrix_dw = []
        test_feature_matrix_n2v = []
        test_feature_matrix_lap = []
        test_feature_matrix_hope = []
        test_label_vector = []

        for num in range(len(test_row)):
            feature_sdne_gene = np.array(gene_emb_matrix_sdne[test_row[num], :])
            feature_sdne_dis = np.array(dis_emb_matrix_sdne[test_col[num], :])
            feature_gf_gene = np.array(gene_emb_matrix_gf[test_row[num], :])
            feature_gf_dis = np.array(dis_emb_matrix_gf[test_col[num], :])
            feature_dw_gene = np.array(gene_emb_matrix_dw[test_row[num], :])
            feature_dw_dis = np.array(dis_emb_matrix_dw[test_col[num], :])
            feature_n2v_gene=np.array(gene_emb_matrix_n2v[test_row[num], :])
            feature_n2v_dis=np.array(dis_emb_matrix_n2v[test_col[num], :])
            feature_lap_gene = np.array(gene_emb_matrix_lap[test_row[num], :])
            feature_lap_dis = np.array(dis_emb_matrix_lap[test_col[num], :])
            feature_hope_gene = np.array(gene_emb_matrix_hope[test_row[num], :])
            feature_hope_dis = np.array(dis_emb_matrix_hope[test_col[num], :])
            

            feature_vector_sdne =np.append(feature_sdne_gene, feature_sdne_dis)
            feature_vector_gf =np.append(feature_gf_gene, feature_gf_dis)
            feature_vector_dw =np.append(feature_dw_gene, feature_dw_dis)
            feature_vector_n2v =np.append(feature_n2v_gene, feature_n2v_dis)
            feature_vector_lap =np.append(feature_lap_gene, feature_lap_dis)
            feature_vector_hope =np.append(feature_hope_gene, feature_hope_dis)
            test_feature_matrix_sdne.append(feature_vector_sdne)
            test_feature_matrix_gf.append(feature_vector_gf)
            test_feature_matrix_dw.append(feature_vector_dw)
            test_feature_matrix_n2v.append(feature_vector_n2v)
            test_feature_matrix_lap.append(feature_vector_lap)
            test_feature_matrix_hope.append(feature_vector_hope)
            test_label_vector.append(gene_dis_matrix[test_row[num], test_col[num]])
            

        train_feature_matrix_sdne = np.array(train_feature_matrix_sdne)
        train_feature_matrix_gf = np.array(train_feature_matrix_gf)
        train_feature_matrix_dw= np.array(train_feature_matrix_dw)
        train_feature_matrix_n2v = np.array(train_feature_matrix_n2v)
        train_feature_matrix_lap = np.array(train_feature_matrix_lap)
        train_feature_matrix_hope = np.array(train_feature_matrix_hope)
        train_label_vector = np.array(train_label_vector)
        test_feature_matrix_sdne = np.array(test_feature_matrix_sdne)
        test_feature_matrix_gf = np.array(test_feature_matrix_gf)
        test_feature_matrix_dw= np.array(test_feature_matrix_dw)
        test_feature_matrix_n2v = np.array(test_feature_matrix_n2v)
        test_feature_matrix_lap = np.array(test_feature_matrix_lap)
        test_feature_matrix_hope = np.array(test_feature_matrix_hope)
        test_label_vector = np.array(test_label_vector)
		
        sdne_prob1=make_prediction1(train_feature_matrix_sdne,train_label_vector,test_feature_matrix_sdne)
        gf_prob1=make_prediction1(train_feature_matrix_gf,train_label_vector,test_feature_matrix_gf)
        dw_prob1=make_prediction1(train_feature_matrix_dw,train_label_vector,test_feature_matrix_dw)
        n2v_prob1=make_prediction1(train_feature_matrix_n2v,train_label_vector,test_feature_matrix_n2v)
        lap_prob1=make_prediction1(train_feature_matrix_lap,train_label_vector,test_feature_matrix_lap)
        hope_prob1=make_prediction1(train_feature_matrix_hope,train_label_vector,test_feature_matrix_hope)
        sdne_prob2 = make_prediction2(train_feature_matrix_sdne, train_label_vector, test_feature_matrix_sdne)
        gf_prob2 = make_prediction2(train_feature_matrix_gf, train_label_vector, test_feature_matrix_gf)
        dw_prob2 = make_prediction2(train_feature_matrix_dw, train_label_vector, test_feature_matrix_dw)
        n2v_prob2 = make_prediction2(train_feature_matrix_n2v, train_label_vector, test_feature_matrix_n2v)
        lap_prob2 = make_prediction2(train_feature_matrix_lap, train_label_vector, test_feature_matrix_lap)
        hope_prob2 = make_prediction2(train_feature_matrix_hope, train_label_vector, test_feature_matrix_hope)
        
        mul_prob =sdne_prob2+gf_prob2+dw_prob2+n2v_prob2+lap_prob2+hope_prob2+sdne_prob1+gf_prob1+dw_prob1+n2v_prob1+lap_prob1+hope_prob1
        all_pred_proba=mul_prob/6
        
        vec = []
        for i in range(len(all_pred_proba)):
            if (all_pred_proba[i] > 0.5):
                vec.append(1)
            else:
                vec.append(0)
        all_pred_label=np.array(vec)
        ensemble_results=ensemble_scoring(test_label_vector, all_pred_label,all_pred_proba)
        print(ensemble_results)
        metric_csv.append(ensemble_results)
        metric += ensemble_results

    metric = metric / kf.n_splits
    print(metric)

    metric = np.array(metric)
    name = 'seed=' + str(seed) + '.csv'
    np.savetxt(name, metric, delimiter=',')
    return metric

if __name__=="__main__":
    gene_dis_matrix = np.loadtxt('data/gene-dis.csv', delimiter=',', dtype=int)
    dis_chemical_matrix = np.loadtxt('data/dis_chem.csv', delimiter=',', dtype=int)
    gene_chemical_matrix = np.loadtxt('data/chem-gene.csv', delimiter=',', dtype=int)
    gene_chemical_matrix=np.transpose(gene_chemical_matrix)
    gene_gene_matrix = np.loadtxt('data/gene-gene-network.csv', delimiter=',', dtype=int)
    result=np.zeros((1,7),float)
    average_result=np.zeros((1,7),float)
    circle_time=5

    for i in range(circle_time):
        result+=cross_validation_experiment(gene_dis_matrix,dis_chemical_matrix,gene_chemical_matrix,gene_gene_matrix,i,1)

    average_result=result/circle_time
    print(average_result)

