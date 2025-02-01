#while rerunning the code make sure all the ML Model folders are renamed or deleted because it will overwrite the exsisitng !!!!
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import glob
import joblib
import warnings
import os
import threading
import logging
import time
import math
from math import sqrt
import datetime
import warnings
from numpy import array
from keras.models import Sequential,Model,load_model,save_model
from keras.layers import Activation,Dropout, Dense
from keras.layers import Flatten,LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import glob
import sys
from keras.layers import SimpleRNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from multiprocessing import Process
from sklearn import preprocessing
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
logging.basicConfig(filename="ML_Process_Log.txt", level=logging.DEBUG, format="%(asctime)s %(message)s", filemode="w")
import psutil
from tqdm import tqdm
from time import sleep
import psutil
import logging
import pywt
from sklearn.preprocessing import MinMaxScaler

cwd=os.getcwd()
st_all= time.time()
class preprocess:
  def specs(self,name):
    global cwd_mem
    with open(cwd_mem+"usage.txt", "a") as f:
      f.write("\n")
      f.write(name)
      f.write("\n")
      f.write("########################")
      f.write("\n")
      cpu_start=psutil.cpu_percent()
      f.write('Total CPUs utilized percentage:')
      f.write(str(cpu_start))
      f.write("\n")
      per=psutil.virtual_memory().percent
      f.write('Total RAM utilized percentage:')
      f.write(str(per))
      f.write("\n")
      f.write("Available memory (Gb):")
      mem_aval=psutil.virtual_memory()[1]
      mem_aval=mem_aval/1024/1024/1024
      f.write(str(mem_aval))
      f.write("\n")
      f.write("##########################")
 
  def dwt_train(self):
    os.chdir(cwd+"/train/")
    files=cwd+"/train/"
    file_path=files
    os.makedirs(cwd+"/train_wav/",exist_ok=True)
    t1=glob.glob(file_path+"*.csv")
    new_data=pd.DataFrame()
    global wavelet_level
    global wavelet_mode
    global wavelet_type
    print("Wavelet preprocessing!!!!")
    wavelet_type = input("enter the wavelet type :")
    wavelet_mode = input("enter the wavelet mode :")
    wavelet_level= int(input("enter the level of wavlet :"))
    print("Converting time domain data into wavelet coefficients of Train files !!!!!")   
    for file in t1:
        cols=[]
        filename=file.replace(file_path,"") #read file from path into induvidual files
        filename1= filename.replace(".csv","")
        df=pd.read_csv(file)
        scaler = MinMaxScaler()
        del df['Sim_time']
        proc=df['process']
        volt=df['voltage']
        temp=df['temperature']
        coeffs_dwt=pd.DataFrame()
        cols=df.columns
        cols= [e for e in cols if e not in ('process', 'voltage','temperature','I59.counter[6:0]','I59.us_count[6:0]')]
        df[cols] = scaler.fit_transform(df[cols])
        for col in cols:
          coeffs=pywt.wavedec(df[col],wavelet_type,mode=wavelet_mode,level=wavelet_level)
          slice=pywt.coeffs_to_array(coeffs)[0]
          coeffs_dwt[col]=slice
        coeffs_dwt['process']=proc
        coeffs_dwt['voltage']=volt
        coeffs_dwt['temperature']=temp
        #coeffs_dwt = coeffs_dwt.dropna()
        coeffs_dwt = coeffs_dwt.fillna(method='ffill')
        coeffs_dwt.to_csv(cwd+"/train_wav/"+filename1+".csv",index=False)
    print("Wavelet convertion successful of Train files!!!!!")

  def dwt_test(self):
    os.chdir(cwd+"/test/")
    files=cwd+"/test/"
    file_path=files
    os.makedirs(cwd+"/test_wav/",exist_ok=True)
    t1=glob.glob(file_path+"*.csv")
    new_data=pd.DataFrame()
    print("Converting time domain data into wavelet coefficients of Test files!!!!!") 
    for file in t1:
        cols=[]
        filename=file.replace(file_path,"") #read file from path into induvidual files
        filename1= filename.replace(".csv","")
        df=pd.read_csv(file)
        scaler = MinMaxScaler()
        proc=df['process']
        volt=df['voltage']
        temp=df['temperature']
        time=df['Sim_time']
        del df['Sim_time']
        cols=df.columns
        cols= [e for e in cols if e not in ('process', 'voltage','temperature','I59.us_count[6:0]','I59.counter[6:0]')]
        df[cols] = scaler.fit_transform(df[cols])
        coeffs_dwt=pd.DataFrame()
        for col in cols:
          coeffs=pywt.wavedec(df[col],wavelet_type,mode=wavelet_mode,level=wavelet_level)
          slice=pywt.coeffs_to_array(coeffs)[0]
          coeffs_dwt[col]=slice
        coeffs_dwt['Sim_time']=time
        coeffs_dwt['process']=proc
        coeffs_dwt['voltage']=volt
        coeffs_dwt['temperature']=temp
        coeffs_dwt = coeffs_dwt.fillna(method='ffill')
        coeffs_dwt.to_csv(cwd+"/test_wav/"+filename1+".csv",index=False) 
    print("Wavelet convertion successful for Test files!!!!!")

  def combine_files(self):
    global combined_csv
    st = time.time()
    stime = datetime.datetime.now()
    print("Combining csv start time:-", stime)
    print("Data Processing started ")
    logging.debug("Data Processing......")
    os.chdir(cwd+"/train_wav/")# folder path    
    CHUNK_SIZE = 5000
    extension = 'csv'
    csv_file_list= [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv=cwd+"/combined_train_wavelet.csv"
    first_one = True
    for csv_file_name in csv_file_list:
      if not first_one: # if it is not the first csv file then skip the header row (row 0) of that file
       skip_row = [0]
      else:
       skip_row = []
      chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE, skiprows = skip_row,header=None)
      for chunk in chunk_container:
       chunk.to_csv(combined_csv, mode="a",index=False,header=False)
      first_one = False    
    et = time.time()
    etime = datetime.datetime.now()
    print("combining csv end time:-", etime)
    logging.debug("Data processing completed")
    elapsed_time = et - st
    print('Execution time for combining csv is:', elapsed_time, 'seconds')
    print('Data Processing completed')

  def fetch(self):
    global combined_csv
    print("Fetching Data.....")
    st = time.time()
    stime = datetime.datetime.now()
    print("data fetch start time:-", stime)
    combined_csv=pd.read_csv(cwd+"/"+"combined_train_wavelet.csv",chunksize=5000) # split into chunks rather than storing into memory all at once
    combined_csv=pd.concat(combined_csv)
    #combined_csv['process'] = combined_csv['process'].replace({'Nominal':1,'weak':2,'strong':3,'nominal':1})
    label_encoder = preprocessing.LabelEncoder()
    #combined_csv['process']= label_encoder.fit_transform(combined_csv['process'])
    etime = datetime.datetime.now()
    print("data fetch end time:-", etime)
    print('Data Fetched Sucessfully.....')
    et = time.time()
    elapsed_time = et - st
    print('Execution time for fetching data is:', elapsed_time, 'seconds')
  
class RandomForest:
  global cwd_rf
  global wavelet_level
  global wavelet_mode
  global wavelet_type
  os.makedirs(cwd+"/RandomForest/",exist_ok=True)
  cwd_rf=cwd+"/RandomForest/"
  #model creation using trained data
  def model_creation(self):
    logging.debug("Model Creation started")
    print("Random Forest Model Creation Started.....")
    st = time.time()
    stime = datetime.datetime.now()
    print("Random Forest Model creation start time:-", stime)
    os.makedirs(cwd_rf+"/models",exist_ok=True)
    my_file = open(cwd+"/input_signals.txt", "r")
    lines = [line.strip() for line in my_file]
    #reading output signals
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    inputs=lines
    outputs=lines1
    X=combined_csv[inputs].values
    est=int(input("enter the no of trees for random forest modelling:"))
    label_encoder = preprocessing.LabelEncoder()
    #combined_csv['process']= label_encoder.fit_transform(combined_csv['process'])
    for i in outputs:
      print("Creating model for signal-",i)
      y=combined_csv[i].values
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      rf= RandomForestRegressor(n_estimators =est,n_jobs=-1)# use all cores available for modelling
      print("model fitting")
      model=rf.fit(X_train,y_train)
      print("fitting completed")
      y_train_predicted = model.predict(X_train)
      y_test_predicted = model.predict(X_test)
      mse_train = mean_squared_error(y_train, y_train_predicted)
      mse_test = mean_squared_error(y_test, y_test_predicted)
      print("Random forest, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))
      os.makedirs(cwd_rf+"/models",exist_ok=True)
      model_path=cwd_rf+"/models/signal_"+i+".joblib"
      joblib.dump(model,open(model_path,'wb'),compress=3)
    et = time.time()
    etime = datetime.datetime.now()
    print("Model creation end time:-", etime)
    elapsed_time = et - st
    print('Execution time for creating model is :', elapsed_time, 'seconds')
    logging.debug("model saved sucessfully")
    print("RANDOM FOREST MODEL SAVED SUCESSFULLY")
    
  #prediction of test file using trained model
  def prediction(self):
    st = time.time()
    print("RANDOM FOREST PREDICTION STARTED...")
    logging.debug("prediction started")
    stime = datetime.datetime.now()
    print("Prediction start time:-", stime)
    file_path=cwd+"/test_wav/"
    t1=glob.glob(file_path+"*.csv")
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      label_encoder = preprocessing.LabelEncoder()
      #df['process']= label_encoder.fit_transform(df['process'])
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"/input_signals.txt", "r")
      # reading the file
      lines= [line.strip() for line in my_file] 
      my_file1 = open(cwd+"/output_signals.txt", "r")
      # reading the file
      lines1= [line.strip() for line in my_file1] 
      logging.debug("signals from output file are read sucessfully for prediction")
      inputs=lines
      outputs=lines1
      label_encoder = preprocessing.LabelEncoder()
      #df['process']= label_encoder.fit_transform(df['process'])
      for i in outputs:
       row=df[inputs].values
       model_path=cwd_rf+"/models/signal_"+i+".joblib"
       model=model_path
       regression=joblib.load(open(model,'rb'))
       logging.debug("trained model loaded sucessfully")
       os.makedirs(cwd_rf+"/predictions/"+i,exist_ok=True)
       yhat =regression.predict(row)
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time'] 
       name['predicted_signal']=yhat#acess numpy array colum wise
       name['actual_signal']=df[i]
       name.to_csv(cwd_rf+"/predictions/"+i+"/"+filename1+".csv")
    et = time.time()  
    etime = datetime.datetime.now()
    print("Prediction end time:-", etime)
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")
    
  def idwt(self):
    global wavelet_level
    global wavelet_mode
    global wavelet_type
    global time
    st=time.time()
    logging.debug("IDWT CONVERSION STARTED-RF")
    print("IDWT CONVERSION STARTED [RF] ...")
    def get_csv_files(folder):
      csv_files = []
      for file in os.listdir(folder):
          if file.endswith('.csv'):
              csv_files.append(file)
      return csv_files
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1= [line.strip() for line in my_file1] 
    wavelet_type = input("enter the wavelet type :")
    wavelet_mode = input("enter the wavelet mode :")
    wavelet_level= int(input("enter the level of wavlet :"))
    for i in lines1:
      folder1_path = cwd_rf+"predictions/"+i+"/"
      folder2_path = cwd+"/test/"
      new=pd.DataFrame()
      os.makedirs(cwd_rf+"predictions_idwt/"+i,exist_ok=True)
      files1 = get_csv_files(folder1_path)
      files2 = get_csv_files(folder2_path)
      for file1 in files1:
        if file1 in files2:
            file1_path = os.path.join(folder1_path, file1)
            file2_path = os.path.join(folder2_path, file1)
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                pred= pd.read_csv(f1)
                Time= pd.read_csv(f2)
                actual=Time[i]
                coeffs=pred['predicted_signal']
                sim_time=Time['Sim_time']
                sz =len(Time)
                dummy_signal = [i for i in range(0, sz)]
                coeffs_dummy = pywt.wavedec(dummy_signal,wavelet_type,wavelet_mode,wavelet_level) 
                coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                coeffs_idwt=pywt.array_to_coeffs(coeffs,coeff_slices,output_format='wavedec')
                coeffs_idwt1=pywt.waverec(coeffs_idwt,wavelet_type,wavelet_mode)
                new=pd.DataFrame()
                coeffs_idwt1=coeffs_idwt1[0:sz]
                new['actual_signal']=actual
                new['predicted_signal']=coeffs_idwt1
                new['Sim_time']=sim_time
                new.to_csv(cwd_rf+'predictions_idwt/'+i+"/"+file1)
      et = time.time()
      elapsed_time = et - st
      print('Execution time for IDWT conversion is :', elapsed_time, 'seconds')
      logging.debug("IDWT conversion sucessfull")
      print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCESSFULLY !!!!")
  

  #plotting graph between actual signal and predicetd signal  
  def graph_plot(self):
    st = time.time()
    cwd_rf=cwd+"/RandomForest/"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      path = cwd_rf+"/predictions_idwt/"+i+"/"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_rf+"/graphs/"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal']
       pred_vinn = df_fun['predicted_signal']
       fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
       plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
       plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
       title = name
       plt.xlabel("Time", fontsize=10)
       plt.ylabel("comp_out", fontsize=10)
       plt.grid(True)
       plt.legend()
       plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
       plt.title(title)
       nm =cwd_rf+"/graphs/"+i+"/"+name+".png"
       plt.savefig(nm)
      et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is :', elapsed_time, 'seconds')
    logging.debug("graph plotted sucessfull")
    print("GRAPH PLOTTED SUCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      file_path = cwd_rf+"/predictions_idwt/"+i+"/"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_rf+"/metrics/"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal"] - df["predicted_signal"]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal'].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal']
       vinn=df['actual_signal']
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       result_file =cwd_rf+"/metrics/"+i+"/"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")

#class MLP
class MLP:
#model creation using trained data
  def model_creation(self):
    st = time.time()
    global cwd_mlp
    os.makedirs(cwd+"/mlp/",exist_ok=True)
    cwd_mlp=cwd+"/mlp/"
    print("MLP Model Creation started")
    logging.debug("files in train folder combined for modeling")
    combined_csv=pd.read_csv(cwd+"/combined_train_wavelet.csv",chunksize=50000)
    combined_csv=pd.concat(combined_csv)
    label_encoder = preprocessing.LabelEncoder()
    #combined_csv['process']= label_encoder.fit_transform(combined_csv['process']) 
    my_file = open(cwd+"/input_signals.txt", "r")
    #reading file
    lines = [line.strip() for line in my_file]
    logging.debug("signals from input file are read sucessfully")
    #reading output signals
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1 = [line.strip() for line in my_file1]
    logging.debug("signals from output file are read sucessfully")
    inputs=lines
    X=combined_csv[inputs].values
    outputs=lines1
    for i in outputs:
      os.makedirs(cwd_mlp+"/models/"+i,exist_ok=True)
      y=combined_csv[i].values
      logging.debug("input and output values are assigned to X and y")
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      logging.debug("input,output data has been split for train and test")
      logging.debug("model creation started")
      model = Sequential()
      model.add(Dense(50,activation='relu'))
      model.add(Dense(30,activation='relu'))#hidden layer
      model.add(Dense(1))#output layer
      model.compile(optimizer=Adam(lr=0.001), loss='mse')
      history="history"+i
      history=model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=64,epochs=2)
      logging.debug("model created sucessfully")
      model.save(cwd_mlp+"/models/"+i+"/")
      # summarize history for accuracy and loss
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig(cwd_mlp+"/models/"+i+"/"+"model_graph_"+i+".png")
      et = time.time()
      elapsed_time = et - st
      print('Execution time for creating model is :', elapsed_time, 'seconds')
      logging.debug("model saved sucessfully")
      print("MLP MODEL SAVED SUCESSFULLY")
#prediction of test file using trained model
  def prediction(self):
    st = time.time()
    cwd_mlp=cwd+"/mlp/"
    print("MLP PREDICTION STARTED...")
    logging.debug("prediction started")
    file_path=cwd+"/test_wav/"
    t1=glob.glob(file_path+"*.csv")
    output=pd.DataFrame()
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"/input_signals.txt", "r")
      # reading the file
      lines = [line.strip() for line in my_file]
      my_file1 = open(cwd+"/output_signals.txt", "r")
      # reading the file
      lines1 = [line.strip() for line in my_file1]
      logging.debug("signals from output file are read sucessfully for prediction")
      outputs=lines1
      label_encoder = preprocessing.LabelEncoder()
      #df['process']= label_encoder.fit_transform(df['process'])
      for i in outputs:
       my_model = keras.models.load_model(cwd_mlp+"/models/"+i+"/")
       os.makedirs(cwd_mlp+"/predictions/"+i+"/",exist_ok=True)
       row=df[lines].values
       logging.debug("trained model loaded sucessfully")
       prediction=my_model.predict(row,verbose=0)
       prediction = [item for sublist in prediction for item in sublist]
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal']=prediction#acess numpy array colum wise
       name['actual_signal']=df[i]
       name.to_csv(cwd_mlp+"/predictions/"+i+"/"+filename1+".csv")
    et = time.time()
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")
  
  def idwt(self):
    global time
    st=time.time()
    cwd_mlp=cwd+"/mlp/"
    logging.debug("IDWT CONVERSION STARTED")
    print("IDWT CONVERSION STARTED [MLP]...")
    def get_csv_files(folder):
      csv_files = []
      for file in os.listdir(folder):
          if file.endswith('.csv'):
              csv_files.append(file)
      return csv_files
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      folder1_path = cwd_mlp+"predictions/"+i+"/"
      folder2_path = cwd+"/test/"
      new=pd.DataFrame()
      os.makedirs(cwd_mlp+"predictions_idwt/"+i,exist_ok=True)
      files1 = get_csv_files(folder1_path)
      files2 = get_csv_files(folder2_path)
      wavelet_type = input("enter the wavelet type :")
      wavelet_mode = input("enter the wavelet mode :")
      wavelet_level= int(input("enter the level of wavlet :"))
      for file1 in files1:
        if file1 in files2:
            file1_path = os.path.join(folder1_path, file1)
            file2_path = os.path.join(folder2_path, file1)
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                pred= pd.read_csv(f1)
                Time= pd.read_csv(f2)
                actual=Time[i]
                coeffs=pred['predicted_signal']
                sim_time=Time['Sim_time']
                sz =len(Time)
                dummy_signal = [i for i in range(0, sz)]
                coeffs_dummy = pywt.wavedec(dummy_signal,wavelet_type,wavelet_mode,wavelet_level)
                coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                coeffs_idwt=pywt.array_to_coeffs(coeffs,coeff_slices,output_format='wavedec')
                coeffs_idwt1=pywt.waverec(coeffs_idwt,wavelet_type,wavelet_mode)
                new=pd.DataFrame()
                coeffs_idwt1=coeffs_idwt1[0:sz]
                new['actual_signal']=actual
                new['predicted_signal']=coeffs_idwt1
                new['Sim_time']=sim_time
                new.to_csv(cwd_mlp+'predictions_idwt/'+i+'/'+file1)
      et = time.time()
      elapsed_time = et - st
      print('Execution time for IDWT conversion is :', elapsed_time, 'seconds')
      logging.debug("IDWT conversion sucessfull")
      print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCESSFULLY !!!!")

  #plotting graph between actual signal and predicetd signal  
  def graph_plot(self):
    st = time.time()
    cwd_mlp=cwd+"/mlp/"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      path = cwd_mlp+"/predictions_idwt/"+i+"/"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_mlp+"/graphs/"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal']
       pred_vinn = df_fun['predicted_signal']
       fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
       plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
       plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
       title = name
       plt.xlabel("Time", fontsize=10)
       plt.ylabel("vinn", fontsize=10)
       plt.grid(True)
       plt.legend()
       plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
       plt.title(title)
       nm =cwd_mlp+"/graphs/"+i+"/"+name+".png"
       plt.savefig(nm)
      et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is :', elapsed_time, 'seconds')
    logging.debug("graph plotted sucessfull")
    print("GRAPH PLOTTED SUCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    cwd_mlp=cwd+"/mlp/"
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      file_path = cwd_mlp+"/predictions_idwt/"+i+"/"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_mlp+"/metrics/"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal"] - df["predicted_signal"]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal'].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal']
       vinn=df['actual_signal']
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       result_file =cwd_mlp+"/metrics/"+i+"/"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")

class lstm_new:
  def model_creation(self):
    st = time.time()
    global cwd_lstm
    os.makedirs(cwd+"/lstm/",exist_ok=True)
    cwd_lstm=cwd+"/lstm/"
    print("LSTM Model creation Started")
    os.makedirs(cwd_lstm+"/models",exist_ok=True)
    logging.debug("files in train folder combined for modeling")
    combined_csv=pd.read_csv(cwd+"/combined_train_wavelet.csv",chunksize=50000)
    combined_csv=pd.concat(combined_csv)
    label_encoder = preprocessing.LabelEncoder()
    #combined_csv['process']= label_encoder.fit_transform(combined_csv['process'])
    my_file = open(cwd+"/input_signals.txt", "r")
    #reading file
    lines = [line.strip() for line in my_file]
    logging.debug("signals from input file are read sucessfully")
    #reading output signals
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1 = [line.strip() for line in my_file1]
    logging.debug("signals from output file are read sucessfully")
    inputs=lines
    X=combined_csv[inputs].values
    X= np.asarray(X)
    outputs=lines1
    for i in outputs:
      os.makedirs(cwd_lstm+"/models/"+i+"/",exist_ok=True)
      y=combined_csv[i].values
      y= np.asarray(y)
      length=len(combined_csv)
      in_len=len(inputs)
      X = X.reshape(length,1,in_len)
      y = array(y)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
      model = Sequential()  
      model.add(LSTM(10,activation='relu',return_sequences=True, input_shape = (1,in_len)))
      model.add(LSTM(32,activation='relu'))
      model.add(Dense(1))
      model.compile(optimizer=Adam(lr=0.001), loss='mse')
      history=model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=64,epochs=2)
      model.save(cwd_lstm+"/models/"+i+"/")
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig(cwd_lstm+"/models/"+i+"/"+"model_graph_"+i+".png")
      et = time.time()
      elapsed_time = et - st
      print('Execution time for creating model is :', elapsed_time, 'seconds')
      logging.debug("model saved sucessfully")
      print("LSTM MODEL SAVED SUCESSFULLY")

  def prediction(self):
    st = time.time()
    cwd_lstm=cwd+"/lstm/"
    print("LSTM PREDICTION STARTED...")
    logging.debug("prediction started")
    file_path=cwd+"/test_wav/"
    t1=glob.glob(file_path+"*.csv")
    output=pd.DataFrame()
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"/input_signals.txt", "r")
      # reading the file
      lines = [line.strip() for line in my_file]
      my_file1 = open(cwd+"/output_signals.txt", "r")
      # reading the file
      lines1 = [line.strip() for line in my_file1]
      logging.debug("signals from output file are read sucessfully for prediction")
      outputs=lines1
      inputs=lines
      length=len(df)
      in_len=len(inputs)
      label_encoder = preprocessing.LabelEncoder()
      #df['process']= label_encoder.fit_transform(df['process'])
      for i in outputs:
       my_model = keras.models.load_model(cwd_lstm+"/models/"+i+"/")
       os.makedirs(cwd_lstm+"/predictions/"+i+"/",exist_ok=True)
       row=df[inputs].values
       row = row.reshape(length,1,in_len)
       logging.debug("trained model loaded sucessfully")
       prediction=my_model.predict(row,verbose=0)
       prediction = [item for sublist in prediction for item in sublist]
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal']=prediction #acess numpy array colum wise  
       name['actual_signal']=df[i]
       name.to_csv(cwd_lstm+"/predictions/"+i+"/"+filename1+".csv")
    et = time.time()
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")
  
  def idwt(self):
    global time
    st=time.time()
    logging.debug("IDWT CONVERSION STARTED")
    cwd_lstm=cwd+"/lstm/"
    print("IDWT CONVERSION STARTED [LSTM]...")
    def get_csv_files(folder):
      csv_files = []
      for file in os.listdir(folder):
          if file.endswith('.csv'):
              csv_files.append(file)
      return csv_files
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      folder1_path = cwd_lstm+"predictions/"+i+"/"
      folder2_path = cwd+"/test/"
      new=pd.DataFrame()
      os.makedirs(cwd_lstm+"predictions_idwt/"+i,exist_ok=True)
      files1 = get_csv_files(folder1_path)
      files2 = get_csv_files(folder2_path)
      wavelet_type = input("enter the wavelet type :")
      wavelet_mode = input("enter the wavelet mode :")
      wavelet_level= int(input("enter the level of wavlet :"))
      for file1 in files1:
        if file1 in files2:
            file1_path = os.path.join(folder1_path, file1)
            file2_path = os.path.join(folder2_path, file1)
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                pred= pd.read_csv(f1)
                Time= pd.read_csv(f2)
                actual=Time[i]
                coeffs=pred['predicted_signal']
                sim_time=Time['Sim_time']
                sz =len(Time)
                dummy_signal = [i for i in range(0, sz)] 
                coeffs_dummy = pywt.wavedec(dummy_signal,wavelet_type,wavelet_mode,wavelet_level)
                coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                coeffs_idwt=pywt.array_to_coeffs(coeffs,coeff_slices,output_format='wavedec')
                coeffs_idwt1=pywt.waverec(coeffs_idwt,wavelet_type,wavelet_mode)
                new=pd.DataFrame()
                coeffs_idwt1=coeffs_idwt1[0:sz]
                new['actual_signal']=actual
                new['predicted_signal']=coeffs_idwt1
                new['Sim_time']=sim_time
                new.to_csv(cwd_lstm+'predictions_idwt/'+i+'/'+file1)
      et = time.time()
      elapsed_time = et - st
      print('Execution time for IDWT conversion is :', elapsed_time, 'seconds')
      logging.debug("IDWT conversion sucessfull")
      print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCESSFULLY !!!!")

  def graph_plot(self):
    st = time.time()
    cwd_lstm=cwd+"/lstm/"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      path = cwd_lstm+"/predictions_idwt/"+i+"/"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_lstm+"/graphs/"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal']
       pred_vinn = df_fun['predicted_signal']
       fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
       plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
       plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
       title = name
       plt.xlabel("Time", fontsize=10)
       plt.ylabel("vinn", fontsize=10)
       plt.grid(True)
       plt.legend()
       plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
       plt.title(title)
       nm =cwd_lstm+"/graphs/"+i+"/"+name+".png"
       plt.savefig(nm)
      et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is :', elapsed_time, 'seconds')
    logging.debug("graph plotted sucessfull")
    print("GRAPH PLOTTED SUCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    cwd_lstm=cwd+"/lstm/"
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      file_path = cwd_lstm+"/predictions_idwt/"+i+"/"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_lstm+"/metrics/"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal"] - df["predicted_signal"]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal'].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal']
       vinn=df['actual_signal']
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       result_file =cwd_lstm+"/metrics/"+i+"/"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")
    
if __name__ == "__main__":

  print('''   
                  __  __    _         _         _    _ _ _ _   _ _ _ _ _ 
                 |  \/  |  | |       | |   _   | |  |  _ _ _| |_ _   _ _|
                 | \  / |  | |       | |  / \  | |  | |           | |
                 | |\/| |  | |       | | /   \ | |  | |           | |
                 | |  | |  | |____   | |/ / \ \| |  | |_ _ _      | |
                 |_|  |_|  |______|  |__ /   \ __|  |_ _ _ _|     |_|     \n ''')
  print('''                 *** MACHINE LEARNING BASED AMS WAVEFORM PREDICTION SCRIPT *** \n \n              
	         MACHINE LEARNING AMS CIRCUIT MODELLING USING WAVELET TRANSFORM\n
                 1. RANDOM FOREST MODELLING  \n
                 2. MULTI-LAYER PERCEPTRON MODELLING \n
                 3. LSTM MODELLING \n
                 4. ALL ALGORITHMS MODELLING \n
                 5. EXIT PROGRAM  \n ''')
  choice = int(input("Select an option: "))
  rf= RandomForest()
  multi=MLP()
  lstm=lstm_new()
  pr=preprocess()
  os.makedirs(cwd+"/Memory_usage/",exist_ok=True)
  cwd_mem=cwd+"/Memory_usage/"
  if choice == 1:
    st_all=time.time()
    print("RANDOM FOREST MODELLING SELECTED")
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    pr.specs("DWT conversion process")
    pr.dwt_train()
    pr.dwt_test()
    pr.specs("BEFORE ALL PROCESS")
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES")
    pr.fetch()
    pr.specs("AFTER FETCHING DATA")
    rf.model_creation()
    pr.specs("AFTER MODEL CREATION")
    rf.prediction()
    pr.specs("AFTER PREDICTION")
    rf.idwt()  
    rf.graph_plot()
    rf.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for random forest Modelling is :', elapsed_time, 'seconds')
  elif choice == 2:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##MLP MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("MLP MODELLING SELECTED")
    pr.specs("BEFORE ALL PROCESS-MLP")
    pr.dwt_train()
    pr.dwt_test()
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES-MLP")
    multi.model_creation()
    pr.specs("AFTER MODEL CREATION")
    multi.prediction()
    pr.specs("AFTER PREDICTION")
    multi.idwt()
    multi.graph_plot()
    multi.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for MLP is :', elapsed_time, 'seconds')
  elif choice == 3:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##LSTM MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("LSTM MODELLING SELECTED")
    pr.specs("BEFORE ALL PROCESS-LSTM")
    pr.dwt_train()
    pr.dwt_test()
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES-LSTM")
    pr.fetch()
    lstm.model_creation()
    pr.specs("AFTER MODEL CREATION-LSTM")
    lstm.prediction()
    pr.specs("AFTER PREDICTION-LSTM")
    lstm.idwt()
    lstm.graph_plot()
    lstm.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for LSTM is :', elapsed_time, 'seconds')
  elif choice== 4:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##ALL MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("ALL ML MODELLING SELECTED")
    pr.dwt_train()
    pr.dwt_test()
    pr.specs("BEFORE ALL PROCESS-ALL")
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES-ALL")
    pr.fetch()
    pr.specs("AFTER FETCHING COMBINED FILES-ALL")
    rf.model_creation()
    pr.specs("AFTER MODEL CREATION-RF") 
    multi.model_creation()
    pr.specs("AFTER MODEL CREATION-MLP")
    lstm.model_creation()
    pr.specs("AFTER MODEL CREATION-LSTM")
    rf.prediction()
    pr.specs("AFTER PREDICTION-RF")
    multi.prediction()
    pr.specs("AFTER PREDICTION-MLP")
    lstm.prediction()
    pr.specs("AFTER PREDICTION-LSTM")
    rf.idwt()
    multi.idwt()
    lstm.idwt()
    rf.graph_plot()
    multi.graph_plot()
    lstm.graph_plot() 
    rf.metrics()
    multi.metrics()
    lstm.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for all algorithm is :', elapsed_time, 'seconds')
  elif choice == 5:
    sys.exit()




            
