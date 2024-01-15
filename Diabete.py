import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

from funzioni import *

import warnings
warnings.filterwarnings("ignore")


dataset = pd.read_csv("diabetes_prediction_dataset.csv")
print("Number of null data:\n",dataset.isnull().sum())
print("Number of duplicated data: ",dataset.duplicated().sum())
dataset.drop_duplicates(inplace=True)
print("Number of duplicated data: ",dataset.duplicated().sum())

categorical_data = ['gender','hypertension','heart_disease','smoking_history','diabetes']
numeric_data = ['age','bmi', 'HbA1c_level', 'blood_glucose_level']

plt.figure(figsize = (15,10))
sns.heatmap(dataset.corr(), annot=True)
plt.subplots_adjust(bottom=0.235 , left= 0.259)
plt.show()

print(dataset.info())

#VARIABILI CATEGORICHE
plt.figure(figsize = (15,10))
plt.suptitle("Categorical Features", fontweight='bold', fontsize=18, fontfamily='sans-serif')

plt.subplot(2,3,1)
stringLabel=['Female', 'Male', 'Other']
categoricalDataVisualization(dataset,'gender','Gender',[0,1,2], stringLabel)

plt.subplot(2,3,2)
stringLabel=['No Hypertension', 'Yes Hypertension']
categoricalDataVisualization(dataset,'hypertension','Hypertension',[0,1], stringLabel)

plt.subplot(2,3,3)
stringLabel=['No Heart Disease', 'Yes Heart Disease']
categoricalDataVisualization(dataset,'heart_disease','Heart Disease',[0,1], stringLabel)

plt.subplot(2,3,4)
stringLabel=['No Info', 'Never', 'Former', 'Current', 'Not Current', 'Ever']
categoricalDataVisualization(dataset,'smoking_history','Smoking History',[0,1,2,3,4,5], stringLabel)

plt.subplot(2,3,5)
stringLabel=['No Diabetes', 'Yes Diabetes']
categoricalDataVisualization(dataset,'diabetes','Diabetes',[0,1], stringLabel)

plt.subplots_adjust(wspace=0.265, bottom=0.078 , left= 0.083, right=0.995)
plt.show()

#VARIABILI CONTINUE
plt.figure(figsize=(15,10))
plt.suptitle("Numerical Features", fontweight='bold', fontsize=18, fontfamily='sans-serif')

plt.subplot(2,2,1)
numericDataVisualization(dataset, 'age', 'Age')

plt.subplot(2,2,2)
numericDataVisualization(dataset, 'bmi', 'Body Mass Index')

plt.subplot(2,2,3)
numericDataVisualization(dataset, 'HbA1c_level', 'Hemoglobin A1C')

plt.subplot(2,2,4)
numericDataVisualization(dataset, 'blood_glucose_level', 'Blood Glucose Level')

plt.show()

plt.title("Diabetes Count with Outliers", fontweight='bold', fontsize=18, fontfamily='sans-serif')
stringLabel=['No Diabetes', 'Yes Diabetes']
categoricalDataVisualization(dataset,'diabetes','Diabetes',[0,1], stringLabel)
plt.show()

sns.set(rc={'figure.figsize':(15,10)})
sea_boxplot=sns.boxplot(orient="h",data=dataset)
sea_boxplot.set(xlabel='Valori', ylabel='Feature')
plt.title("Boxplot Features with Outliers", fontweight='bold', fontsize=18, fontfamily='sans-serif')

plt.show()

numeric_columns = (list(dataset.loc[:, 'gender':'diabetes']))
Outliers_IQR = IQR_method(dataset,1,numeric_columns)
df_out = dataset.drop(Outliers_IQR, axis = 0).reset_index(drop=True)

sns.set(rc={'figure.figsize':(15,10)})
sea_boxplot=sns.boxplot(orient="h",data=df_out)
sea_boxplot.set(xlabel='Valori', ylabel='Feature')
plt.title("Boxplot Features without Outliers", fontweight='bold', fontsize=18, fontfamily='sans-serif')

plt.show()

plt.title("Diabetes Count without Outliers", fontweight='bold', fontsize=18, fontfamily='sans-serif')
stringLabel=['No Diabetes', 'Yes Diabetes']
categoricalDataVisualization(df_out,'diabetes','Diabetes',[0,1], stringLabel)
plt.show()

X = dataset.drop('diabetes', axis=1) 
Y = dataset['diabetes']      

scaler = RobustScaler()
     
X_rob = scaler.fit_transform(X)
     
X_train, X_test, y_train, y_test= train_test_split(X_rob, Y, stratify=Y, test_size = 0.3)

#------------------------------------------------------------------------------------------------------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Logistic Regression 

log_reg = LogisticRegression(solver = 'liblinear')
log_reg_score = cross_validate(log_reg, X_train, y_train, cv=skf, scoring='recall')
recall_test = log_reg_score['test_score']
recall_mean = np.mean(log_reg_score['test_score'])

print('\nLogistic Regression')
print("Recall Scores: ", recall_test)
print("Average Recall Score : ", recall_mean)

#K-Nearest Neighbors

best_k =  bestNeighborsNumber(X_train,y_train,'knn')
knn = KNeighborsClassifier(n_neighbors=best_k)
knn_score = cross_validate(knn, X_train, y_train, cv=skf, scoring='recall')
recall_test = knn_score['test_score']
recall_mean = np.mean(knn_score['test_score'])

print('\nK-Nearest Neighbors')
print("Recall Scores: ", recall_test)
print("Average Recall Score : ", recall_mean)

#Decision Tree
dt = DecisionTreeClassifier()
dt_score = cross_validate(dt, X_train, y_train, cv=skf, scoring='recall')
recall_test = dt_score['test_score']
recall_mean = np.mean(dt_score['test_score'])

print('\nDecision Tree')
print("Recall Scores: ", recall_test)
print("Average Recall Score : ", recall_mean)

#RandomForest
rf = RandomForestClassifier()
rf_score = cross_validate(rf, X_train, y_train, cv=skf, scoring = 'recall')
recall_test = rf_score['test_score']
recall_mean = np.mean(rf_score['test_score'])

print('\nRandom Forest')
print("Recall Scores: ", recall_test)
print("Average Recall Score : ", recall_mean)

#------------------------------------------------------------------------------------------------------------------------------------

#LogisticRegression Hyperparameter

param_grid_log_reg = [
    {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty' : ['l2'],'max_iter' : [100,1000,5000]},
    {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' : ['l1'],'max_iter' : [100,1000,5000]}]
    
grid_log_reg = GridSearchCV(log_reg, param_grid=param_grid_log_reg, cv=skf, scoring='recall').fit(X_train, y_train)
     
print('\nBest Parameters Logistic Regression:', grid_log_reg.best_params_)
    
y_pred_grid_log_reg = grid_log_reg.predict(X_test)
     
grid_log_reg_recall = recall_score(y_test, y_pred_grid_log_reg)
grid_log_reg_precision = precision_score(y_test, y_pred_grid_log_reg)
grid_log_reg_f1 = f1_score(y_test, y_pred_grid_log_reg)
grid_log_reg_accuracy = accuracy_score(y_test, y_pred_grid_log_reg)

log_reg_list_hype = [(grid_log_reg_accuracy, grid_log_reg_precision,grid_log_reg_recall, grid_log_reg_f1)]

log_reg_visualization_hype = pd.DataFrame(data=log_reg_list_hype, columns=['Accuracy','Precision','Recall','F1 Score'])
log_reg_visualization_hype.insert(0, 'Model', 'Logistic Regression')
print(log_reg_visualization_hype)

confusionMatrix(y_test, y_pred_grid_log_reg,'Logistic Regression')

#K-Nearest Neighbors Hyperparameter
  
param_grid_knn= {
    'weights':['uniform','distance'],
    'algorithm':['auto','ball_tree','kd_tree','brute'],
    'leaf_size':[10,20,30,40,50],
    'p':[1,2],
    'metric': ['euclidean','manhattan','minkowski'],
}
  
grid_knn = GridSearchCV(knn, param_grid=param_grid_knn, cv=skf, scoring='recall').fit(X_train, y_train)
     
print('\nBest Parameters K-Nearest Neighbors:', grid_knn.best_params_)
   
y_pred_grid_knn = grid_knn.predict(X_test)
     
grid_knn_recall = recall_score(y_test, y_pred_grid_knn)
grid_knn_precision = precision_score(y_test, y_pred_grid_knn)
grid_knn_f1 = f1_score(y_test, y_pred_grid_knn)
grid_knn_accuracy = accuracy_score(y_test, y_pred_grid_knn)

knn_list_hype = [(grid_knn_accuracy, grid_knn_precision, grid_knn_recall, grid_knn_f1)]

knn_visualization_hype = pd.DataFrame(data=knn_list_hype, columns=['Accuracy','Precision','Recall','F1 Score'])
knn_visualization_hype.insert(0, 'Model', 'K-Nearest Neighbors')
print(knn_visualization_hype)

confusionMatrix(y_test, y_pred_grid_knn,'K-Nearest Neighbors')

#DecisionTree Hyperparameter

param_grid_dt = {
    'criterion' : ['gini', 'entropy','log_loss'],
    'splitter': ['best','random'],
    'max_depth': [10, 20, 30, 40, 50],       
    'min_samples_split': [2, 5, 10, 20],     
    'min_samples_leaf': [1, 2, 4, 7]
}

grid_dt = GridSearchCV(dt, param_grid=param_grid_dt, cv=skf, scoring='recall').fit(X_train, y_train)
     
print('\nBest Parameters Decision Tree:', grid_dt.best_params_)
    
y_pred_grid_dt = grid_dt.predict(X_test)
     
grid_dt_recall = recall_score(y_test, y_pred_grid_dt)
grid_dt_precision = precision_score(y_test, y_pred_grid_dt)
grid_dt_f1 = f1_score(y_test, y_pred_grid_dt)
grid_dt_accuracy = accuracy_score(y_test, y_pred_grid_dt)

dt_list_hype = [(grid_dt_accuracy, grid_dt_precision, grid_dt_recall, grid_dt_f1)]

dt_visualization_hype = pd.DataFrame(data=dt_list_hype, columns=['Accuracy','Precision','Recall','F1 Score'])
dt_visualization_hype.insert(0, 'Model', 'Decision Tree')
print(dt_visualization_hype)

confusionMatrix(y_test, y_pred_grid_dt,'Decision Tree')

#RandomForest Hyperparameter

param_grid_rf = { 'criterion' : ['gini', 'entropy','log_loss'],
              'n_estimators': [25, 50, 75, 100, 150, 200, 250] }

grid_rf = GridSearchCV(rf,param_grid=param_grid_rf, cv=skf, scoring='recall').fit(X_train, y_train)
     
print('\nBest Parameters Random Forest:', grid_rf.best_params_)
    
y_pred_grid_rf = grid_rf.predict(X_test)
     
grid_rf_recall = recall_score(y_test, y_pred_grid_rf)
grid_rf_precision = precision_score(y_test, y_pred_grid_rf)
grid_rf_f1 = f1_score(y_test, y_pred_grid_rf)
grid_rf_accuracy = accuracy_score(y_test, y_pred_grid_rf)

rf_list_hype = [(grid_rf_recall, grid_rf_precision, grid_rf_f1, grid_rf_accuracy)]

rf_visualization_hype = pd.DataFrame(data=rf_list_hype, columns=['Accuracy','Precision','Recall','F1 Score'])
rf_visualization_hype.insert(0, 'Model', 'Random Forest')
print(rf_visualization_hype)
print('\n')

confusionMatrix(y_test, y_pred_grid_rf,'Random Forest')

model_visualization_hype = pd.concat([log_reg_visualization_hype, knn_visualization_hype, dt_visualization_hype, rf_visualization_hype], ignore_index=True, sort=False)
model_visualization_hype.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_visualization_hype)
print('\n')

rocCurveComparison(grid_log_reg,grid_knn,grid_dt,grid_rf, X_test,y_test, "Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest")

#------------------------------------------------------------------------------------------------------------------------------------

#LogisticRegression Balanced

log_reg_bal=LogisticRegression(solver = 'liblinear', class_weight= 'balanced')
grid_log_reg_bal = GridSearchCV(log_reg_bal, param_grid=param_grid_log_reg, cv=skf, scoring='recall').fit(X_train, y_train)
print('\nBest Parameters Logistic Regression Balanced:', grid_log_reg_bal.best_params_)

y_pred_log_reg_bal = grid_log_reg_bal.predict(X_test)

log_reg_recall_bal = recall_score(y_test, y_pred_log_reg_bal)
log_reg_precision_bal = precision_score(y_test, y_pred_log_reg_bal)
log_reg_f1_bal = f1_score(y_test, y_pred_log_reg_bal)
log_reg_accuracy_bal = accuracy_score(y_test, y_pred_log_reg_bal)

log_reg_list_bal = [(log_reg_accuracy_bal, log_reg_precision_bal, log_reg_recall_bal, log_reg_f1_bal)]

log_reg_visualization_bal = pd.DataFrame(data=log_reg_list_bal, columns=['Accuracy','Precision','Recall','F1 Score'])
log_reg_visualization_bal.insert(0, 'Model', 'Logistic Regression Balanced')
print('\nLogistic Regression Balanced')
print(log_reg_visualization_bal)

confusionMatrix(y_test, y_pred_log_reg_bal,'Logistic Regression Balanced')

#K-nearest Neighbors Balanced
best_k =  bestNeighborsNumber(X_train, y_train,'knnBal')
knn_bal = KNeighborsClassifier(n_neighbors=best_k,  weights='uniform')
grid_knn_bal = GridSearchCV(knn_bal, param_grid=param_grid_knn, cv=skf, scoring='recall').fit(X_train, y_train)
print('\nBest Parameters K-Nearest Neighbors Balanced:', grid_knn_bal.best_params_)

y_pred_knn_bal = grid_knn_bal.predict(X_test)

knn_bal_recall = recall_score(y_test, y_pred_knn_bal)
knn_bal_precision = precision_score(y_test, y_pred_knn_bal)
knn_bal_f1 = f1_score(y_test, y_pred_knn_bal)
knn_bal_accuracy = accuracy_score(y_test, y_pred_knn_bal)

knn_list_bal = [(knn_bal_accuracy, knn_bal_precision, knn_bal_recall, knn_bal_f1)]

knn_visualization_bal = pd.DataFrame(data=knn_list_bal, columns=['Accuracy','Recall','Precision','F1 Score'])
knn_visualization_bal.insert(0, 'Model', 'K-Nearest Neighbors Balanced')
print('\nK-Nearest Neighbors Balanced')
print(knn_visualization_bal)

confusionMatrix(y_test, y_pred_knn_bal,'K-Nearest Neighbors Balanced')

#DecisionTree Balanced
dt_bal=DecisionTreeClassifier(class_weight='balanced')
grid_dt_bal = GridSearchCV(dt_bal, param_grid=param_grid_dt, cv=skf, scoring='recall').fit(X_train, y_train)
print('\nBest Parameters Decision Tree Balanced:', grid_dt_bal.best_params_)

y_pred_dt_bal = grid_dt_bal.predict(X_test)

dt_bal_recall = recall_score(y_test, y_pred_dt_bal)
dt_bal_precision = precision_score(y_test, y_pred_dt_bal)
dt_bal_f1 = f1_score(y_test, y_pred_dt_bal)
dt_bal_accuracy = accuracy_score(y_test, y_pred_dt_bal)

dt_list_bal = [(dt_bal_accuracy, dt_bal_precision,dt_bal_recall, dt_bal_f1)]

dt_visualization_bal = pd.DataFrame(data=dt_list_bal, columns=['Accuracy','Precision','Recall','F1 Score'])
dt_visualization_bal.insert(0, 'Model', 'Decision Tree Balanced')
print('\nDecision Tree Balanced')
print(dt_visualization_bal)

confusionMatrix(y_test, y_pred_dt_bal,'Decision Tree Balanced')

#RandomForest Balanced
rf_bal = RandomForestClassifier(class_weight='balanced')
grid_rf_bal = GridSearchCV(rf_bal,param_grid=param_grid_rf, cv=skf, scoring='recall').fit(X_train, y_train)
print('\nBest Parameters Random Forest Balanced:', grid_rf_bal.best_params_)

y_pred_rf_bal = grid_rf_bal.predict(X_test)

rf_bal_recall = recall_score(y_test,y_pred_rf_bal)
rf_bal_precision = precision_score(y_test, y_pred_rf_bal)
rf_bal_f1 = f1_score(y_test, y_pred_rf_bal)
rf_bal_accuracy = accuracy_score(y_test, y_pred_rf_bal)

rf_list_bal = [(rf_bal_accuracy, rf_bal_precision, rf_bal_recall, rf_bal_f1)]

rf_visualization_bal = pd.DataFrame(data=rf_list_bal, columns=['Accuracy','Precision', 'Recall','F1 Score'])
rf_visualization_bal.insert(0, 'Model', 'Random Forest Balanced')
print('\nRandom Forest Balanced')
print(rf_visualization_bal)

confusionMatrix(y_test, y_pred_rf_bal,'Random Forest Balanced')


model_visualization_bal = pd.concat([log_reg_visualization_bal, knn_visualization_bal, dt_visualization_bal, rf_visualization_bal], ignore_index=True, sort=False)
model_visualization_bal.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_visualization_bal)
print('\n')

rocCurveComparison(grid_log_reg_bal,grid_knn_bal,grid_dt_bal,grid_rf_bal, X_test,y_test, "Logistic Regression Balanced", "K-Nearest Neighbors Balanced", "Decision Tree Balanced", "Random Forest Balanced")

#------------------------------------------------------------------------------------------------------------------------------------

#Logistic Regression Oversampling

print("\nPrima del random oversampling con SMOTE:")
print("Classe 0 no-diabetico:", np.sum(Y == 0))
print("Classe 1 diabetico:", np.sum(Y == 1))

random_oversampler = RandomOverSampler(random_state=42)
X_oversampled, y_oversampled = random_oversampler.fit_resample(X_train, y_train)

smote = SMOTE( random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_oversampled, y_oversampled)

print("\nDopo il random oversampling con SMOTE:")
print("Classe 0 (no-diabete) dopo SMOTE:", np.sum(y_train_resampled == 0))
print("Classe 1 (diabete) dopo SMOTE:", np.sum(y_train_resampled == 1))

log_reg_over=LogisticRegression(solver='liblinear')
grid_log_reg_over= GridSearchCV(log_reg_over, param_grid=param_grid_log_reg, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters Logistic Regression Oversampling:', grid_log_reg_over.best_params_)

y_pred_log_reg_over = grid_log_reg_over.predict(X_test)

over_log_reg_recall = recall_score(y_test, y_pred_log_reg_over)
over_log_reg_precision = precision_score(y_test, y_pred_log_reg_over)
over_log_reg_f1 = f1_score(y_test, y_pred_log_reg_over)
over_log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg_over)

log_reg_list_over = [(over_log_reg_accuracy, over_log_reg_precision,over_log_reg_recall, over_log_reg_f1)]

log_reg_visualization_over = pd.DataFrame(data=log_reg_list_over, columns=['Accuracy','Precision','Recall','F1 Score'])
log_reg_visualization_over.insert(0, 'Model', 'Logistic Regression Oversampling')
print('\nLogistic Regression Oversampling')
print(log_reg_visualization_over)

confusionMatrix(y_test, y_pred_log_reg_over,'Logistic Regression Oversampling')

#K-nearest Neighbors Oversampling
best_k =  bestNeighborsNumber(X_train_resampled, y_train_resampled,'knnOver')
knn_over = KNeighborsClassifier(n_neighbors=best_k)

grid_knn_over = GridSearchCV(knn_over, param_grid=param_grid_knn, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters K-Nearest Neighbors Oversampling:', grid_knn_over.best_params_)

y_pred_knn_over = grid_knn_over.predict(X_test)

knn_over_recall = recall_score(y_test, y_pred_knn_over)
knn_over_precision = precision_score(y_test, y_pred_knn_over)
knn_over_f1 = f1_score(y_test, y_pred_knn_over)
knn_over_accuracy = accuracy_score(y_test, y_pred_knn_over)

knn_list_over = [(knn_over_accuracy, knn_over_precision, knn_over_recall, knn_over_f1)]

knn_visualization_over = pd.DataFrame(data=knn_list_over, columns=['Accuracy','Recall','Precision','F1 Score'])
knn_visualization_over.insert(0, 'Model', 'K-Nearest Neighbors Oversampling')
print('\nK-Nearest Neighbors Oversampling')
print(knn_visualization_over)

confusionMatrix(y_test, y_pred_knn_over,'K-Nearest Neighbors Oversampling')

#DecisionTree Oversampling
dt_over=DecisionTreeClassifier()
grid_dt_over = GridSearchCV(dt_over, param_grid=param_grid_dt, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters Decision Tree Oversampling:', grid_dt_over.best_params_)

y_pred_dt_over = grid_dt_over.predict(X_test)

dt_over_recall = recall_score(y_test, y_pred_dt_over)
dt_over_precision = precision_score(y_test, y_pred_dt_over)
dt_over_f1 = f1_score(y_test, y_pred_dt_over)
dt_over_accuracy = accuracy_score(y_test, y_pred_dt_over)

dt_list_over = [(dt_over_accuracy, dt_over_precision,dt_over_recall, dt_over_f1)]

dt_visualization_over = pd.DataFrame(data=dt_list_over, columns=['Accuracy','Precision','Recall','F1 Score'])
dt_visualization_over.insert(0, 'Model', 'Decision Tree Oversampling')
print('\nDecision Tree Oversampling')
print(dt_visualization_over)

confusionMatrix(y_test, y_pred_dt_over,'Decision Tree Oversampling')

#RandomForest Oversampling
rf_over = RandomForestClassifier()
grid_rf_over = GridSearchCV(rf_over,param_grid=param_grid_rf, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters Random Forest Oversampling:', grid_rf_over.best_params_)

y_pred_rf_over = grid_rf_over.predict(X_test)

rf_over_recall = recall_score(y_test,y_pred_rf_over)
rf_over_precision = precision_score(y_test, y_pred_rf_over)
rf_over_f1 = f1_score(y_test, y_pred_rf_over)
rf_over_accuracy = accuracy_score(y_test, y_pred_rf_over)

rf_list_over = [(rf_over_accuracy, rf_over_precision, rf_over_recall, rf_over_f1)]

rf_visualization_over = pd.DataFrame(data=rf_list_over, columns=['Accuracy','Precision', 'Recall','F1 Score'])
rf_visualization_over.insert(0, 'Model', 'Random Forest Oversampling')
print('\nRandom Forest Oversampling')
print(rf_visualization_over)

confusionMatrix(y_test, y_pred_rf_over,'Random Forest Oversampling')

model_visualization_over = pd.concat([log_reg_visualization_over, knn_visualization_over, dt_visualization_over, rf_visualization_over], ignore_index=True, sort=False)
model_visualization_over.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_visualization_over)
print('\n')

rocCurveComparison(grid_log_reg_over,grid_knn_over,grid_dt_over,grid_rf_over, X_test,y_test, "Logistic Regression Oversampling", "K-Nearest Neighbors Oversampling", "Decision Tree Oversampling", "Random Forest Oversampling")

#------------------------------------------------------------------------------------------------------------------------------------

#Undersampling

print("\nPrima l'undersampling")
print("Classe 0 no-diabetico:", np.sum(Y == 0))
print("Classe 1 diabetico:", np.sum(Y == 1))

random_undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = random_undersampler.fit_resample(X_train, y_train)

print("\nDopo l'undersampling")
print("Classe 0 no-diabetico:", np.sum(y_train_resampled == 0))
print("Classe 1 diabetico:", np.sum(y_train_resampled == 1))


#Logistic Regression Undersampling

log_reg_under=LogisticRegression(solver='liblinear')
grid_log_reg_under= GridSearchCV(log_reg_under, param_grid=param_grid_log_reg, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters Logistic Regression Undersampling:', grid_log_reg_under.best_params_)

y_pred_log_reg_under = grid_log_reg_under.predict(X_test)

under_log_reg_recall = recall_score(y_test, y_pred_log_reg_under)
under_log_reg_precision = precision_score(y_test, y_pred_log_reg_under)
under_log_reg_f1 = f1_score(y_test, y_pred_log_reg_under)
under_log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg_under)

log_reg_list_under = [(under_log_reg_accuracy, under_log_reg_precision,under_log_reg_recall, under_log_reg_f1)]

log_reg_visualization_under = pd.DataFrame(data=log_reg_list_under, columns=['Accuracy','Precision','Recall','F1 Score'])
log_reg_visualization_under.insert(0, 'Model', 'Logistic Regression Undersampling')
print('\nLogistic Regression Undersampling')
print(log_reg_visualization_under)

confusionMatrix(y_test, y_pred_log_reg_under,'Logistic Regression Undersampling')

#K-nearest Neighbors Undersampling
best_k =  bestNeighborsNumber(X_train_resampled, y_train_resampled,'knnUnder')
knn_under = KNeighborsClassifier(n_neighbors=best_k)
grid_knn_under= GridSearchCV(knn_under, param_grid=param_grid_knn, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters K-Nearest Neighbors Undersampling:', grid_knn_under.best_params_)

y_pred_knn_under = grid_knn_under.predict(X_test)

knn_under_recall = recall_score(y_test, y_pred_knn_under)
knn_under_precision = precision_score(y_test, y_pred_knn_under)
knn_under_f1 = f1_score(y_test, y_pred_knn_under)
knn_under_accuracy = accuracy_score(y_test, y_pred_knn_under)

knn_list_under = [(knn_under_accuracy, knn_under_precision, knn_under_recall, knn_under_f1)]

knn_visualization_under = pd.DataFrame(data=knn_list_under, columns=['Accuracy','Recall','Precision','F1 Score'])
knn_visualization_under.insert(0, 'Model', 'K-Nearest Neighbors Undersampling')
print('\nK-Nearest Neighbors Undersampling')
print(knn_visualization_under)

confusionMatrix(y_test, y_pred_knn_under,'K-Nearest Neighbors Undersampling')

#DecisionTree Undersampling
dt_under=DecisionTreeClassifier()
grid_dt_under = GridSearchCV(dt_under, param_grid=param_grid_dt, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters Decision Tree Undersampling:', grid_dt_under.best_params_)

y_pred_dt_under = grid_dt_under.predict(X_test)

dt_under_recall = recall_score(y_test, y_pred_dt_under)
dt_under_precision = precision_score(y_test, y_pred_dt_under)
dt_under_f1 = f1_score(y_test, y_pred_dt_under)
dt_under_accuracy = accuracy_score(y_test, y_pred_dt_under)

dt_list_under = [(dt_under_accuracy, dt_under_precision,dt_under_recall, dt_under_f1)]

dt_visualization_under = pd.DataFrame(data=dt_list_under, columns=['Accuracy','Precision','Recall','F1 Score'])
dt_visualization_under.insert(0, 'Model', 'Decision Tree Undersampling')
print('\nDecision Tree Undersampling')
print(dt_visualization_under)

confusionMatrix(y_test, y_pred_dt_under,'Decision Tree Undersampling')

#RandomForest Undersampling
rf_under = RandomForestClassifier()
grid_rf_under = GridSearchCV(rf_under,param_grid=param_grid_rf, cv=skf, scoring='recall').fit(X_train_resampled, y_train_resampled)
print('\nBest Parameters Random Forest Undersampling:', grid_rf_under.best_params_)

y_pred_rf_under = grid_rf_under.predict(X_test)

rf_under_recall = recall_score(y_test,y_pred_rf_under)
rf_under_precision = precision_score(y_test, y_pred_rf_under)
rf_under_f1 = f1_score(y_test, y_pred_rf_under)
rf_under_accuracy = accuracy_score(y_test, y_pred_rf_under)

rf_list_under = [(rf_under_accuracy, rf_under_precision, rf_under_recall, rf_under_f1)]

rf_visualization_under = pd.DataFrame(data=rf_list_under, columns=['Accuracy','Precision', 'Recall','F1 Score'])
rf_visualization_under.insert(0, 'Model', 'Random Forest Undersampling')
print('\nRandom Forest Undersampling')
print(rf_visualization_under)

confusionMatrix(y_test, y_pred_rf_under,'Random Forest Undersampling')

model_visualization_under = pd.concat([log_reg_visualization_under, knn_visualization_under, dt_visualization_under, rf_visualization_under], ignore_index=True, sort=False)
model_visualization_under.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_visualization_under)
print('\n')

rocCurveComparison(grid_log_reg_under,grid_knn_under,grid_dt_under,grid_rf_under, X_test,y_test, "Logistic Regression Undersampling", "K-Nearest Neighbors Undersampling", "Decision Tree Undersampling", "Random Forest Undersampling")

#----------------------------------------------------------------------------------------------------------------------------------------

print('\nLogistic Regression - Logistic Regression Balanced - Logistic Regression Oversampling - Logistic Regression Undersampling\n')
model_lg_visualization_bal = pd.concat([log_reg_visualization_hype,log_reg_visualization_bal,
                                        log_reg_visualization_over,log_reg_visualization_under], ignore_index=True, sort=False)
model_lg_visualization_bal.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_lg_visualization_bal)

rocCurveComparison(grid_log_reg, grid_log_reg_bal,grid_log_reg_over, grid_log_reg_under,  X_test, y_test, 
                   'Logistic Regression', 'Logistic Regression Balanced',
                   'Logistic Regression Oversampling', 'Logistic Regression Undersampling')

print('\nK-nearest Neighbors - K-nearest Neighbors Balanced - K-nearest Neighbors Oversampling - K-nearest Neighbors Undersampling\n')
model_knn_visualization_bal = pd.concat([knn_visualization_hype,knn_visualization_bal,
                                        knn_visualization_over,knn_visualization_under], ignore_index=True, sort=False)
model_knn_visualization_bal.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_knn_visualization_bal)

rocCurveComparison(grid_knn, grid_knn_bal,grid_knn_over, grid_knn_under,  X_test, y_test, 
                  'K-nearest Neighbors', 'K-nearest Neighbors Balanced',
                  'K-nearest Neighbors Oversampling', 'K-nearest Neighbors Undersampling')

print('\nDecision Tree - Decision Tree Balanced - Decision Tree Oversampling - Decision Tree Undersampling\n')
model_dt_visualization_bal = pd.concat([dt_visualization_hype,dt_visualization_bal,
                                        dt_visualization_over,dt_visualization_under], ignore_index=True, sort=False)
model_dt_visualization_bal.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_dt_visualization_bal)

rocCurveComparison(grid_dt, grid_dt_bal, grid_dt_over, grid_dt_under,  X_test, y_test, 
                  'Decision Tree', 'Decision Tree Balanced',
                  'Decision Tree Oversampling', 'Decision Tree Undersampling')

print('\nRandom Forest - Random Forest Balanced - Random Forest Oversampling - Random Forest Undersampling\n')
model_rf_visualization_bal = pd.concat([rf_visualization_hype,rf_visualization_bal,
                                        rf_visualization_over,rf_visualization_under], ignore_index=True, sort=False)
model_rf_visualization_bal.sort_values(by=['Recall'], ascending=False, inplace=True)
print(model_rf_visualization_bal)

rocCurveComparison(grid_rf, grid_rf_bal, grid_rf_over, grid_rf_under,  X_test, y_test, 
                  'Random Forest', 'Random Forest Balanced',
                  'Random Forest Oversampling', 'Random Forest Undersampling')

#------------------------------------------------------------------------------------------------------------------------------------


#Apprendimento non supervisionato 

k = clusterNumber(X_rob)

kmeans_model = KMeans(n_clusters = k)
kmeans_model.fit(X_rob)
y_pred = kmeans_model.predict(X_test)

clusteringVisualization(X,kmeans_model)

X_pca = X_rob.copy()

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_pca)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_reduced)
labels = kmeans.labels_
pcaKmeans(X_reduced, labels)

#------------------------------------------------------------------------------------------------------------------------------------

#Rete Bayesiana
df_RBayes = pd.DataFrame(np.array(dataset.copy(), dtype=int), columns=dataset.columns)

k2 = K2Score(df_RBayes)
hc_k2 = HillClimbSearch(df_RBayes)
modello_k2 = hc_k2.estimate(scoring_method=k2)

print(modello_k2.nodes())  

print(modello_k2.edges())

rete_bayesiana = BayesianNetwork(modello_k2.edges())
rete_bayesiana.fit(df_RBayes)

inferenza = VariableElimination(rete_bayesiana)
prob_notdb = inferenza.query(variables=['diabetes'],
                              evidence={'gender': 0, 'age': 80.0, 'hypertension': 0,
                                         'heart_disease': 1, 'smoking_history': 1,'bmi': 15,
                                        'HbA1c_level': 6, 'blood_glucose_level': 140})
 
print('\nDati nuovo individuo:\n'+
      'gender: 0,\nage: 80.0,\nhypertension: 0,\n'+
      'heart_disease: 1,\nsmoking_history: 1,\nbmi: 15.06\n'+
      'HbA1c_level: 6,\nblood_glucose_level: 140\n\n'+
      'Probabilità per l\'individuo di non avere il diabete: ')
print(prob_notdb, '\n')
 
prob_db = inferenza.query(variables=['diabetes'],
                              evidence={'gender': 1, 'age': 67.0, 'hypertension': 0,
                                         'heart_disease': 1, 'smoking_history': 4,'bmi': 27,
                                        'HbA1c_level': 6, 'blood_glucose_level': 200})
 
print('\nDati nuovo individuo:\n'+
      'gender: 1,\nage: 67.0,\nhypertension: 0,\n'+
      'heart_disease: 1,\nsmoking_history: 4,\nbmi:27.32\n'+
      'HbA1c_level: 6,\nblood_glucose_level: 200\n\n'+
      'Probabilità per l\'individuo di avere il diabete: ')
print(prob_db, '\n\n')

true_labels = df_RBayes['diabetes']

df = df_RBayes.drop(columns=['diabetes'] )

predicted_labels = rete_bayesiana.predict(df)

confusionMatrix(true_labels, predicted_labels, 'Bayesian Network')

print("\nClassification Report Bayesian Network:\n", classification_report(true_labels, predicted_labels))

#------------------------------------------------------------------------------------------------------------------------------------
