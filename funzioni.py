from collections import Counter
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

black_grad = ['#100C07', '#3E3B39', '#6D6A6A', '#9B9A9C', '#CAC9CD']
color_palette=['#ff1104','#bcdaab','#ffe0b5','#ff8b37','#6f004c','#0078F9']

def confusionMatrix(y_test, y_pred, modelName):
  cm = confusion_matrix(y_test,y_pred)
  c_matrix=pd.DataFrame(cm,columns=['Diabetes', 'No Diabetes'],index=['Diabetes', 'No Diabetes'])

  fig,ax=plt.subplots(figsize=(10,10))
  sns.set(font_scale=1.4)
  sns.heatmap(c_matrix,annot=True,fmt = 'd')
  ax.set_title(modelName,fontsize=20)
  ax.set_xlabel("Predicted",fontsize=20)
  ax.set_ylabel("Actual",fontsize=20)
  plt.show()


def categoricalDataVisualization(dataset,column,nameFeature,numberLabels,stringLabel):
       
    # --- Setting Colors, Labels, Order ---
    colors=color_palette
   
    ax = sns.countplot(x=column, data=dataset, palette= colors,
                    edgecolor=black_grad[2], alpha=0.85)
    for rect in ax.patches:
        ax.text (rect.get_x()+rect.get_width()/2,
                rect.get_height()+4.25,rect.get_height(),
                horizontalalignment='center', fontsize=8,
                bbox=dict(facecolor='none', edgecolor=black_grad[0],
                        linewidth=0.25, boxstyle='round'))
    plt.xlabel(nameFeature, fontweight='bold', fontsize=11, fontfamily='sans-serif',
            color=black_grad[1])
    plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif',
            color=black_grad[1])
    plt.xticks(numberLabels, fontsize = 8)
    plt.grid(axis='y', alpha=0.4)
    plt.legend(stringLabel, fontsize='8',
          title_fontsize='9', loc='upper right', frameon=True)
    


def numericDataVisualization(dataset, nameFeature, title):
    # --- Variable, Color & Plot Size ---
    var = nameFeature
    color_boxPlot = color_palette[0]
    
    # --- Box Plot ---
    sns.boxplot(data=dataset, y=var, color=color_boxPlot, boxprops=dict(alpha=0.8), linewidth=1.5)
    plt.ylabel(title, fontweight='regular', fontsize=11, fontfamily='sans-serif', 
            color=black_grad[2])


def rocCurveComparison(model1, model2,model3, model4, X_test, y_test, modelName1, modelName2,modelName3, modelName4):

        #Model1
        y_scores1 = model1.predict_proba(X_test)[:, 1]
        fpr1, tpr1, thresholds = roc_curve(y_test, y_scores1)
        roc_auc1 = roc_auc_score(y_test, y_scores1)

        #Model2
        y_scores2 = model2.predict_proba(X_test)[:, 1]
        fpr2, tpr2, thresholds = roc_curve(y_test, y_scores2)
        roc_auc2 = roc_auc_score(y_test, y_scores2)

        #Model3
        y_scores3 = model3.predict_proba(X_test)[:, 1]
        fpr3, tpr3, thresholds = roc_curve(y_test, y_scores3)
        roc_auc3 = roc_auc_score(y_test, y_scores3)

        #Model4
        y_scores4 = model4.predict_proba(X_test)[:, 1]
        fpr4, tpr4, thresholds = roc_curve(y_test, y_scores4)
        roc_auc4 = roc_auc_score(y_test, y_scores4)

        #sns.set_style('whitegrid')
        plt.figure(figsize=(15,10))
        plt.title('ROC Curve')
        plt.plot(fpr1,tpr1, lw = 2, color='red', label=modelName1 + f' ROC Curve (AUC = {roc_auc1:.2f})' )
        plt.plot(fpr2,tpr2, lw = 2, color='purple', label=modelName2+ f' ROC Curve (AUC = {roc_auc2:.2f})' )
        plt.plot(fpr3,tpr3, lw = 2, color='green', label=modelName3 + f' ROC Curve (AUC = {roc_auc3:.2f})' )
        plt.plot(fpr4,tpr4, lw = 2, color='blue', label=modelName4 + f' ROC Curve (AUC = {roc_auc4:.2f})' )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(' ROC Curve')
        plt.legend(loc='lower right')
        plt.legend()
        plt.show()

def IQR_method (df,n,features):
    outlier_list = []
   
    for column in features:
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step )].index
        
        outlier_list.extend(outlier_list_column)
        
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )
   
    out1 = df[df[column] < Q1 - outlier_step]
    out2 = df[df[column] > Q3 + outlier_step]
   
    print('Total number of deleted outliers is:', out1.shape[0]+out2.shape[0])

    return multiple_outliers


def clusterNumber(X_rob):
        k_to_test = range(2,25,1) # [2,3,4, ..., 24]
        silhouette_scores = {} 
        for k in k_to_test:
             model_kmeans_k = KMeans( n_clusters = k, n_init='auto')
             model_kmeans_k.fit(X_rob)
             labels_k = model_kmeans_k.labels_
             score_k = metrics.silhouette_score(X_rob, labels_k)
             silhouette_scores[k] = score_k
        
        k = max(silhouette_scores, key=silhouette_scores.get)
        print('Number of cluster:',k)

        plt.figure(figsize = (16,5))
        plt.plot(silhouette_scores.values())
        plt.xticks(range(0,23,1), silhouette_scores.keys())
        plt.title("Silhouette Metric",fontweight='bold', fontsize=18, fontfamily='sans-serif')
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.axvline((k-2), color = "r")
        plt.show()
        return k 

def clusteringVisualization(X,kmeans_model):
        customcmap = ListedColormap([ "red","yellow","silver","blue","purple", "green"])

        fig= plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.scatter(X.values[:, 1], X.values[:, 5], X.values[:, 7],         
                    c=kmeans_model.labels_.astype(float), 
                    edgecolor="k", s=50, cmap=customcmap)
        ax1.view_init(20, -50)
        ax1.set_xlabel('age', fontsize=12)
        ax1.set_ylabel('bmi', fontsize=12)
        ax1.set_zlabel('blood_glucose_level', fontsize=12)
        ax1.set_title("K-Means Clusters", fontweight='bold', fontsize=18, fontfamily='sans-serif')

        fig.show()
        plt.show()

def pcaKmeans(X_reduced, labels):

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['red', 'blue', 'green']
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],  c=[colors[lbl] for lbl in labels])
        ax.set_title("KMeans Clustering (PCA reduced data)",fontweight='bold', fontsize=18, fontfamily='sans-serif')
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        ax.set_zlabel("Third Principal Component")
        
        plt.show()

def bestNeighborsNumber(X_train,y_train, modelName):
        k_values = [i for i in range (1,20)]
        scores = []

        for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                score = cross_val_score(knn, X_train, y_train, cv = 5)
                scores.append(np.mean(score))
        
        plt.plot(k_values, scores, color='g')
        plt.xticks(ticks=k_values, labels=k_values)
        plt.title("Number Of Neighbors")
        plt.grid()
        
        plt.show()

        best_index = np.argmax(scores)
        print('Best Number of Neighbours ' + modelName +':', k_values[best_index])
        return k_values[best_index]