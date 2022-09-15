import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
import itertools
import os
import cv2
import matplotlib.cm


mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH'] 

   

def svmout(file):
    file = pd.read_csv(file)
    df=file
    mean=df.describe()
    
    mean.to_csv("mean.csv") 
    mean=pd.read_csv("mean.csv", index_col=0)
    df = df.drop(['V8','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28'], axis=1)
    
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    print('y : ',Counter(y))
    
    # under sampling
    from imblearn.under_sampling import RandomUnderSampler
    resampled = RandomUnderSampler(sampling_strategy=0.85, random_state=2018)
    X_res, y_res = resampled.fit_resample(X, y)
    print('y_res : ',Counter(y_res))
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_res)
    X = scaler.transform(X_res)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_res, random_state=42)
    
    
##########################################################################3    
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    param = {
        'max_depth' : 3,
        'eta' : 0.1,
        'learning_rate' : 0.01,
        'silent' : 0,
        'objective' : 'multi:softmax',
        'booster' : 'gbtree',
        'n_estimators' : 100,
        'num_class' : 2
        }
    
    bst = xgb.train(param, dtrain, num_boost_round=500)
    train_pred = bst.predict(dtrain)
    test_pred = bst.predict(dtest)
    
    from sklearn.metrics import accuracy_score
    
    Train_Score='Train Set Score : ', accuracy_score(y_train, train_pred)
       
    Test_Score= 'Test Set Score : ', accuracy_score(y_test, test_pred)
     
    models={svm:'SVM'}
    
    f, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,4))
    ax1.hist(df.Time[df.Class==1], bins=50)
    ax1.set_title('Fraud')

    ax2.hist(df.Time[df.Class==0], bins=50)
    ax2.set_title('Normal')

    plt.xlabel('Time(in Seconds)'); plt.ylabel('Number of Transactions')
    plt.title('Simple_plot')
     
    plt.savefig('uploads/Simple_plot.png')
    
    for model in models.keys():
        print(models.get(model),' score : ',model.score(X_test, y_test))
        
        
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        print(cm)
        target_names = ['class 0', 'class 1']

        report = classification_report(y_pred=y_test, y_true=pred,target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).T
        report_df
        report_df.to_csv("report.csv") 
        report=pd.read_csv("report.csv", index_col=0)
        
        
        return (report,mean,Train_Score,Test_Score)
     