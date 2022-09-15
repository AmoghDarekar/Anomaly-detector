import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score,f1_score
import os
from flask import request, jsonify


mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH'] 

  
 
def ALLRESULT(RFC,SVM,LGR,DECT,MLPN,KMC,BRFCA,BBCA):

    df=pd.read_csv('data/file.csv')
    ACC="Accuracy for Classification Report" 
    df = df.drop(['V8','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28'], axis=1)
    
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    
    # under sampling
    from imblearn.under_sampling import RandomUnderSampler
    resampled = RandomUnderSampler(sampling_strategy=0.85, random_state=2018)
    X_res, y_res = resampled.fit_resample(X, y)
    
    
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
    if BBCA == "BBCA":
        from imblearn.ensemble import BalancedBaggingClassifier 
        bbc = BalancedBaggingClassifier(random_state=42)
        bbc.fit(X_train, y_train)
        
        BBCmodels={bbc:'BalancedBaggingClassifier'}
        
        
        for model in BBCmodels.keys():
            BBCM=(BBCmodels.get(model),' score : ',model.score(X_test, y_test))
            
            
        BBCM_out=model.score(X_test, y_test)
        ####################################################################################
        
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        bbc = BalancedBaggingClassifier(random_state=42)
        bbc.fit(X_train, y_train)
        y_pred = bbc.predict(X_test)
        y_pred_proba = bbc.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        
        plt.title("BalancedBaggingClassifier")
        plt.savefig('uploads/01.png')
        ################################################################################
        

        
    else:
        BBCM_out="Not selected for Comparision"
        
############################################################################
    if BRFCA == "BRFCA":
        from imblearn.ensemble import BalancedRandomForestClassifier
        brfc=BalancedRandomForestClassifier()
        brfc.fit(X_train, y_train)
        
        BRFCmodels={brfc:'BalancedRandomForestClassifier'}
            
        for model in BRFCmodels.keys():
            BRFCM=(BRFCmodels.get(model),' score : ',model.score(X_test, y_test))
            
        BRFCM_out=model.score(X_test, y_test)
        ####################################################################################

        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        brfc=BalancedRandomForestClassifier()
        brfc.fit(X_train, y_train)
        y_pred = brfc.predict(X_test)
        y_pred_proba = brfc.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        
        plt.title("BalancedRandomForestClassifier")
        plt.savefig('uploads/02.png')
        ################################################################################

    else:
        BRFCM_out="Not selected for Comparision"
############################################################################
    if LGR == "LGR":  
        from sklearn.linear_model import LogisticRegression
        log = LogisticRegression(C=0.01)
        log.fit(X_train, y_train)
        
        LRCmodels={log:'Logistic Regression'}
        
        for model in LRCmodels.keys():
            LRCM=(LRCmodels.get(model),' score : ',model.score(X_test, y_test))
            
        LRCM_out=model.score(X_test, y_test)
        
        ####################################################################################
        
 
        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        log = LogisticRegression(C=0.01)
        log.fit(X_train, y_train)
        y_pred = log.predict(X_test)
        y_pred_proba = log.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        
        plt.title("Logistic Regression")
        plt.savefig('uploads/03.png')
        ################################################################################
        
    else:
        LRCM_out="Not selected for Comparision"
        
############################################################################
    if DECT == "DECT":
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        
        DTCModels={dt:'Decision Tree'}
        
        for model in DTCModels.keys():
            DTCM=(DTCModels.get(model),' score : ',model.score(X_test, y_test))
            
        DTCM_out=model.score(X_test, y_test)
        
        ####################################################################################

        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        y_pred_proba = dt.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        
        plt.title("Decision Tree")
        plt.savefig('uploads/04.png')
        ################################################################################
        
    else:
        DTCM_out="Not selected for Comparision"
        
############################################################################
    if RFC == "RFC":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        
        RFCmodels={ rf:'Random Forest'}
        
        for model in RFCmodels.keys():
            RFCM=(RFCmodels.get(model),' score : ',model.score(X_test, y_test))
            
        RFCM_out=model.score(X_test, y_test)
        
        ####################################################################################
        

        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        
        plt.title("Random Forest Classifier")
        plt.savefig('uploads/05.png')
        ################################################################################

    else:
        RFCM_out="Not selected for Comparision"        
        
############################################################################
    if SVM == "SVM":
        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(X_train, y_train)
        
        SVCmodels={svm:'SVM'}
        
        for model in SVCmodels.keys():
            SVCM=(SVCmodels.get(model),' score : ',model.score(X_test, y_test))
            
        SVCM_out=model.score(X_test, y_test)
        ####################################################################################
        

        import sklearn.metrics as metrics
        from sklearn.multiclass import OneVsRestClassifier
        svm = SVC( probability=True)
        svm.fit(X_train, y_train)
        # calculate the fpr and tpr for all thresholds of the classification
        
        y_pred = svm.predict(X_test)
        y_pred_proba = svm.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        
        plt.title("SVM Classifier")
        plt.savefig('uploads/06.png')
        ################################################################################

    else:
        SVCM_out="Not selected for Comparision"
       
############################################################################
    if MLPN == "MLPN":
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        
        MLPCmodels={mlp:'Multi layer perceptron'}
        
        
        for model in MLPCmodels.keys():
            MLPCM=(MLPCmodels.get(model),' score : ',model.score(X_test, y_test))
            
        MLPCM_out=(model.score(X_test, y_test))
        ####################################################################################
        
        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        y_pred_proba = mlp.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        
        plt.title("MLP Classifier")
        plt.savefig('uploads/07.png')
        ################################################################################

    else:
        MLPCM_out="Not selected for Comparision"
        
############################################################################
    if KMC == "KMC":
        file = df
        credit_data=file
        
        credit_data =credit_data[credit_data['Amount']==0]
        
        X = credit_data.drop(['Class','Amount'], axis=1)
        y = credit_data['Class']
        
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        sc.fit(X)
        X_sc = sc.transform(X)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(X_sc)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(X_sc)
        X_pca = pca.transform(X_sc)
        
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=0)
        km.fit(X_sc)
        pred = km.predict(X_sc)
        
        target_names = ['class 0', 'class 1']
        report = classification_report(y_pred=pred, y_true=y,target_names=target_names, output_dict=True)
        
        KMEAN_out=f1_score(pred, y, average="macro")
        
        ####################################################################################
        import mglearn
        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        plt.scatter(X_pca[:,0], X_pca[:,1], c=pred, cmap='Paired', s=60, edgecolors='white')
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
                    marker='*',c=[mglearn.cm2(0), mglearn.cm2(1)],s=60, linewidth=2, edgecolor='red')
        plt.xlabel('feature 0'); plt.ylabel('feature 1')
        plt.title('K Mean Clustring Scattered Plot')
        plt.savefig('uploads/08.png')
        ################################################################################
        
    else:
        KMEAN_out="Not selected for Comparision"
        
        
    
#################################################################################################

    models={"List of Classification Model":ACC,'Balanced Bagging Classifier':BBCM_out,"Balanced Random Forest Classifier":BRFCM_out,"Logistic Regression":LRCM_out,
            "Decision Tree Classifier":DTCM_out,"KMeans Clustring":KMEAN_out,"MLP Classifier":MLPCM_out,
            "SVM Classifier":SVCM_out,"RandomForestClassifier":RFCM_out}    
    
    import csv
    with open('data/dict.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in models.items():
           writer.writerow([key, value])
    
    means = pd.read_csv('data/dict.csv')
    

    
    return (means)
    
    
    
    
    
    
    
    
    
    