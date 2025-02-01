
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
import glob
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import pyplot
from multiprocessing import Process
from sklearn import preprocessing
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
logging.basicConfig(filename="ML_Process_Log.txt", level=logging.DEBUG, format="%(asctime)s %(message)s", filemode="w")
import psutil
from tqdm import tqdm
from time import sleep
from keras.callbacks import EarlyStopping
import psutil
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cwd=os.getcwd()
st_all= time.time()

class preprocess:
  def specs(self,name):
    os.makedirs(cwd+"\\RandomForest\\",exist_ok=True)
    cwd_rf=cwd+"\\RandomForest\\"
    with open(cwd_rf+"usage.txt", "a") as f:
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
  def combine_files(self):
    global combined_csv
    cwd=os.getcwd()
    st = time.time()
    os.makedirs(cwd+"\\RandomForest\\",exist_ok=True)
    cwd_rf=cwd+"\\RandomForest\\"
    stime = datetime.datetime.now()
    print("Combining csv start time:-", stime)
    print("Data Processing started ")
    logging.debug("Data Processing......")
    os.chdir(cwd+"\\train\\")# 
    CHUNK_SIZE = 5000
    extension = 'csv'
    csv_file_list= [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv=cwd+"\\combined_train.csv"
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
    print("CSV files combined successfully!")
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
    combined_csv=pd.read_csv(cwd+"\\"+"combined_train.csv",chunksize=5000) # split into chunks rather than storing into memory all at once
    combined_csv=pd.concat(combined_csv)
    #del combined_csv['Unnamed:0']
    #combined_csv.round({"clk_in":5, "fb_clk":5, "Qup":5, "Qdn":5,"Qupb":5,"Qdnb":5,"vco_in":5,"Pll_out":5})
    #combined_csv['process'] = combined_csv['process'].replace({'fastnfastp':1,'fastnslowp':2,'slownslowp':3,'slownfastp':4,'typical':5})
    label_encoder = preprocessing.LabelEncoder()
    #combined_csv['process']= label_encoder.fit_transform(combined_csv['process'])
    #print(combined_csv['process'])
    etime = datetime.datetime.now()
    print("data fetch end time:-", etime)
    print('Data Fetched Sucessfully.....')
    et = time.time()
    elapsed_time = et - st
    print('Execution time for fetching data is:', elapsed_time, 'seconds')
class RandomForest:
  global cwd_rf
  os.makedirs(cwd+"\\RandomForest\\",exist_ok=True)
  cwd_rf=cwd+"\\RandomForest\\" 
  #model creation using trained data
  def model_creation(self):
    logging.debug("Model Creation started")
    print("Random Forest Model Creation Started.....")
    st = time.time()
    stime = datetime.datetime.now()
    print("Random Forest Model creation start time:-", stime)
    os.makedirs(cwd_rf+"\\models",exist_ok=True)
    my_file = open(cwd+"\\input_signals.txt", "r")
    lines = [line.strip() for line in my_file]
    #reading output signals
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    inputs=lines
    outputs=lines1
    X=combined_csv[inputs].values
    for i in outputs:
      print("Creating model for signal-",i)
      y=combined_csv[i].values
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      rf= RandomForestRegressor(n_estimators=10,n_jobs=-1)# use all cores available for modelling
      print("model fitting")
      model=rf.fit(X_train,y_train)
      print("fitting completed")
      y_train_predicted = model.predict(X_train)
      y_test_predicted = model.predict(X_test)
      mse_train = mean_squared_error(y_train, y_train_predicted)
      mse_test = mean_squared_error(y_test, y_test_predicted)
      print("Random forest, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))
      os.makedirs(cwd_rf+"\\models",exist_ok=True) 
      model_path=cwd_rf+"\\models\\signal_"+i+".joblib"
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
    file_path=cwd+"\\test\\"
    t1=glob.glob(file_path+"*.csv")
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      label_encoder = preprocessing.LabelEncoder()
      df['process']= label_encoder.fit_transform(df['process'])
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"\\input_signals.txt", "r")
      # reading the file
      lines= [line.strip() for line in my_file] 
      my_file1 = open(cwd+"\\output_signals.txt", "r")
      # reading the file
      lines1= [line.strip() for line in my_file1] 
      logging.debug("signals from output file are read sucessfully for prediction")
      inputs=lines
      outputs=lines1
      model_path=cwd_rf+"\\models\\signal_"
      for i in outputs:
       row=df[inputs].values
       model_path=cwd_rf+"\\models\\signal_"+i+".joblib"
       model=model_path
       regression=joblib.load(open(model,'rb'))
       logging.debug("trained model loaded sucessfully")
       os.makedirs(cwd_rf+"\\predictions\\"+i,exist_ok=True)
       yhat =regression.predict(row)
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal_'+i]=yhat#acess numpy array colum wise
       name['actual_signal_'+i]=df[i]
       name.to_csv(cwd_rf+"\\predictions\\"+i+"\\"+filename1+"_"+i+".csv")
    et = time.time()  
    etime = datetime.datetime.now()
    print("Prediction end time:-", etime)
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")

  #plotting graph between actual signal and predicetd signal  
  def graph_plot(self):
    st = time.time()
    cwd_rf=cwd+"\\RandomForest\\"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      path = cwd_rf+"\\predictions\\"+i+"\\"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_rf+"\\graphs\\"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal_'+i]
       pred_vinn = df_fun['predicted_signal_'+i]
       fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
       plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
       plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
       title = name
       plt.xlabel("Time", fontsize=10)
       plt.ylabel(i, fontsize=10)
       plt.grid(True)
       plt.legend()
       plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
       plt.title(title)
       nm =cwd_rf+"\\graphs\\"+i+"\\"+name+".png"
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
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      file_path = cwd_rf+"\\predictions\\"+i+"\\"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_rf+"\\metrics\\"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal_"+i] - df["predicted_signal_"+i]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal_'+i].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal_'+i]
       vinn=df['actual_signal_'+i]
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       new_data = {'FILENAME': filename1,'MAE': mae,'MSE': mse,'RMSE': rmse,'R2SCORE': r2,'SNR': snr}
       new_row = pd.DataFrame([new_data])
       result = pd.concat([result, new_row], ignore_index=True)
       result_file =cwd_rf+"\\metrics\\"+i+"\\"+"metrics"+".csv"
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
    os.makedirs(cwd+"\\mlp\\",exist_ok=True)
    cwd_mlp=cwd+"\\mlp\\"
    print("MLP Model Creation started")
    logging.debug("files in train folder combined for modeling")
    combined_csv=pd.read_csv(cwd+"\\combined_train.csv",chunksize=5000000)
    combined_csv=pd.concat(combined_csv)
    my_file = open(cwd+"\\input_signals.txt", "r")
    #reading file
    lines = [line.strip() for line in my_file]
    logging.debug("signals from input file are read sucessfully")
    #reading output signals
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    # reading the file
    lines1 = [line.strip() for line in my_file1]
    logging.debug("signals from output file are read sucessfully")
    inputs=lines
    label_encoder = preprocessing.LabelEncoder()
    combined_csv['process']= label_encoder.fit_transform(combined_csv['process'])
    X=combined_csv[inputs].values
    outputs=lines1
    for i in outputs:
      os.makedirs(cwd_mlp+"\\models\\"+i,exist_ok=True)
      y=combined_csv[i].values
      logging.debug("input and output values are assigned to X and y")
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      logging.debug("input,output data has been split for train and test")
      logging.debug("model creation started")
      num_layers = int(input("Enter the number of hidden layers: "))
    # Step 2: Get the number of neurons for each layer and activation functions
      num_neurons = []
      activations = []
      for j in range(num_layers):
            neurons = int(input(f"Enter the number of neurons for layer {j+1}: "))
            activation = input(f"Enter the activation function for layer {j+1} (e.g., relu, sigmoid, tanh): ")
            num_neurons.append(neurons)
            activations.append(activation)
    # Step 3: Get the learning rate, epochs, and batch size
      learning_rate = float(input("Enter the learning rate (e.g., 0.001, 0.01): "))
      epochs = int(input("Enter the number of epochs (e.g., 10, 50, 100): "))
      batch_size = int(input("Enter the batch size (e.g., 32, 64): "))
      model = Sequential()
      model.add(Dense(num_neurons[0], activation=activations[0]))  # Adjust input_dim as needed
      for k in range(1, num_layers):
           model.add(Dense(num_neurons[k], activation=activations[k]))
      model.add(Dense(1, activation='linear'))
      opt = keras.optimizers.Adam(learning_rate=learning_rate) 
      model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
      history="history"+i
      history=model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=batch_size,epochs=epochs)
      logging.debug("model created sucessfully")
      model.save(cwd_mlp+"\\models\\"+i+"\\")
      # summarize history for accuracy and loss
      pyplot.plot(history.history['loss' ])
      pyplot.plot(history.history['val_loss'])
      pyplot.title('model train vs validation loss')
      pyplot.ylabel('loss')
      pyplot.xlabel('epoch')  
      pyplot.legend(['train' , 'validation' ], loc= 'upper right')
      print("Train vs Validation plotted")
      pyplot.savefig(cwd_mlp+"\\models\\"+i+"\\"+"model_graph_"+i+".png")
      et = time.time()
      elapsed_time = et - st
      print('Execution time for creating model is :', elapsed_time, 'seconds')
      logging.debug("model saved sucessfully")
      print("MLP MODEL SAVED SUCESSFULLY")
#prediction of test file using trained model
  def prediction(self):
    st = time.time()
    cwd_mlp=cwd+"\\mlp\\"
    print("MLP PREDICTION STARTED...")
    logging.debug("prediction started")
    file_path=cwd+"\\test\\"
    t1=glob.glob(file_path+"*.csv")
    output=pd.DataFrame()
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"\\input_signals.txt", "r")
      # reading the file
      lines = [line.strip() for line in my_file]
      my_file1 = open(cwd+"\\output_signals.txt", "r")
      # reading the file
      lines1 = [line.strip() for line in my_file1]
      logging.debug("signals from output file are read sucessfully for prediction")
      outputs=lines1
      label_encoder = preprocessing.LabelEncoder()
      df['process']= label_encoder.fit_transform(df['process'])
      for i in outputs:
       my_model = keras.models.load_model(cwd_mlp+"\\models\\"+i+"\\")
       os.makedirs(cwd_mlp+"\\predictions\\"+i+"\\",exist_ok=True)
       row=df[lines].values
       logging.debug("trained model loaded sucessfully")
       prediction=my_model.predict(row,verbose=0)
       prediction = [item for sublist in prediction for item in sublist]
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal_'+i]=prediction#acess numpy array colum wise
       name['actual_signal_'+i]=df[i]
       name.to_csv(cwd_mlp+"\\predictions\\"+i+"\\ "+filename1+"_"+i+".csv")
    et = time.time()
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")
    
  #plotting graph between actual signal and predicetd signal  
  def graph_plot(self):
    st = time.time()
    cwd_mlp=cwd+"\\mlp\\"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      path = cwd_mlp+"\\predictions\\"+i+"\\"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_mlp+"\\graphs\\"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal_'+i]
       pred_vinn = df_fun['predicted_signal_'+i]
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
       nm =cwd_mlp+"\\graphs\\"+i+"\\"+name+".png"  
       plt.savefig(nm)
      et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is :', elapsed_time, 'seconds')
    logging.debug("graph plotted sucessfull")
    print("GRAPH PLOTTED SUCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    cwd_mlp=cwd+"\\mlp\\"
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      file_path = cwd_mlp+"\\predictions\\"+i+"\\"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns=['FILENAME', 'MAE', 'MSE', 'RMSE', 'R2SCORE', 'SNR'])
      #new_data = {'FILENAME': filename1, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2SCORE': r2, 'SNR': snr}
      os.makedirs(cwd_mlp+"\\metrics\\"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal_"+i] - df["predicted_signal_"+i]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal_'+i].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal_'+i]
       vinn=df['actual_signal_'+i]
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       new_data = {'FILENAME': filename1,'MAE': mae,'MSE': mse,'RMSE': rmse,'R2SCORE': r2,'SNR': snr}
       new_row = pd.DataFrame([new_data])
       result = pd.concat([result, new_row], ignore_index=True)
       result_file =cwd_mlp+"\\metrics\\"+i+"\\"+"metrics"+".csv"
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
    os.makedirs(cwd+"\\lstm\\",exist_ok=True)
    cwd_lstm=cwd+"\\lstm\\"
    print("LSTM Model creation Started")
    os.makedirs(cwd_lstm+"\\models",exist_ok=True)
    logging.debug("files in train folder combined for modeling")
    combined_csv=pd.read_csv(cwd+"\\combined_train.csv",chunksize=50000)
    combined_csv=pd.concat(combined_csv)
    label_encoder = preprocessing.LabelEncoder()
    combined_csv['process']= label_encoder.fit_transform(combined_csv['process'])
    my_file = open(cwd+"\\input_signals.txt", "r")
    #reading file
    lines = [line.strip() for line in my_file]
    logging.debug("signals from input file are read sucessfully")
    #reading output signals
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    # reading the file
    lines1 = [line.strip() for line in my_file1]
    logging.debug("signals from output file are read sucessfully")
    inputs=lines
    X=combined_csv[inputs].values
    X= np.asarray(X)
    outputs=lines1
    for i in outputs:
      os.makedirs(cwd_lstm+"\\models\\"+i+"\\",exist_ok=True)
      y=combined_csv[i].values
      y= np.asarray(y)
      length=len(combined_csv)
      in_len=len(inputs)
      X = X.reshape(length,1,in_len)
      y = array(y)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
      model = Sequential()
      model.add(LSTM(25,activation='relu',return_sequences=True, input_shape = (1,in_len)))
      model.add(LSTM(50,activation='relu'))
      model.add(Dense(1))
      model.compile(optimizer='adam',loss='mse')
      history=model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=32,epochs=2)
      model.save(cwd_lstm+"\\models\\"+i+"\\")
      # summarize history for accuracy and loss
      pyplot.plot(history.history['loss' ])
      pyplot.plot(history.history['val_loss'])
      pyplot.title('model train vs validation loss')
      pyplot.ylabel('loss')
      pyplot.xlabel('epoch')  
      pyplot.legend(['train' , 'validation' ], loc= 'upper right')
      print("Train vs Validation plotted")
      pyplot.savefig(cwd_lstm+"\\models\\"+i+"\\"+"model_graph_"+i+".png")
      et = time.time()
      elapsed_time = et - st
      print('Execution time for creating model is :', elapsed_time, 'seconds')
      logging.debug("model saved sucessfully")
      print("LSTM MODEL SAVED SUCESSFULLY")
  def prediction(self):
    st = time.time()
    cwd_lstm=cwd+"\\lstm\\"
    print("LSTM PREDICTION STARTED...")
    logging.debug("prediction started")
    file_path=cwd+"\\test\\"
    t1=glob.glob(file_path+"*.csv")
    output=pd.DataFrame()
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"\\input_signals.txt", "r")
      # reading the file
      lines = [line.strip() for line in my_file]
      my_file1 = open(cwd+"\\output_signals.txt", "r")
      # reading the file
      lines1 = [line.strip() for line in my_file1]
      logging.debug("signals from output file are read sucessfully for prediction")
      outputs=lines1
      inputs=lines
      length=len(df)
      in_len=len(inputs)
      label_encoder = preprocessing.LabelEncoder()
      df['process']= label_encoder.fit_transform(df['process'])
      for i in outputs:
       my_model = keras.models.load_model(cwd_lstm+"\\models\\"+i+"\\")
       os.makedirs(cwd_lstm+"\\predictions\\"+i+"\\",exist_ok=True)
       row=df[inputs].values
       row = row.reshape(length,1,in_len)
       logging.debug("trained model loaded sucessfully")
       prediction=my_model.predict(row,verbose=0)
       prediction = [item for sublist in prediction for item in sublist]
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal_'+i]=prediction#acess numpy array colum wise
       name['actual_signal_'+i]=df[i]
       name.to_csv(cwd_lstm+"\\predictions\\"+i+"\\"+filename1+"_"+i+".csv")
    et = time.time()
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY") 
  def graph_plot(self):
    st = time.time()
    cwd_lstm=cwd+"\\lstm\\"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      path = cwd_lstm+"\\predictions\\"+i+"\\"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_lstm+"\\graphs\\"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal_'+i]
       pred_vinn = df_fun['predicted_signal_'+i]
       fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
       plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
       plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
       title = name
       plt.xlabel("Time", fontsize=10)
       plt.ylabel(i, fontsize=10)
       plt.grid(True)
       plt.legend()
       plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
       plt.title(title)
       nm =cwd_lstm+"\\graphs\\"+i+"\\"+name+".png"
       plt.savefig(nm)
      et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is :', elapsed_time, 'seconds')
    logging.debug("graph plotted sucessfull")
    print("GRAPH PLOTTED SUCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    cwd_lstm=cwd+"\\lstm\\"
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"\\output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      file_path = cwd_lstm+"\\predictions\\"+i+"\\"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_lstm+"\\metrics\\"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal_"+i] - df["predicted_signal_"+i]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal_'+i].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal_'+i]
       vinn=df['actual_signal_'+i]
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       result_file =cwd_lstm+"\\metrics\\"+i+"\\"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")
    
if __name__ == "__main__":
  print('''   
                  __  __    _         _         _    _ _ _ _   _ _ _ _ _ 
                 |  \\\  |  | |       | |   _   | |  |  _ _ _| |_ _   _ _|
                 | \  \\ |  | |       | |  \\ \  | |  | |           | |
                 | |\\\| |  | |       | | \\   \ | |  | |           | |
                 | |  | |  | |____   | |\\ \\ \ \| |  | |_ _ _      | |
                 |_|  |_|  |______|  |__ \\   \ __|  |_ _ _ _|     |_|     \n ''')
  print('''                 *** MACHINE LEARNING BASED AMS WAVEFORM PREDICTION TOOL *** \n \n              
	         MACHINE LEARNING MODELLING \n
                 1. RANDOM FOREST MODELLING \n
                 2. MULTI-LAYER PERCEPTRON MODELLING \n
                 3. LSTM MODELLING \n
                 4. ALL ALGORITHMS MODELLING \n
                 5. EXIT PROGRAM  \n ''')
  choice = int(input("Select an option: "))
  rf= RandomForest()
  multi=MLP()
  lstm=lstm_new()
  pr=preprocess()
  if choice == 1:
    st_all=time.time()
    print("RANDOM FOREST MODELLING SELECTED") 
    with open(cwd+"usage.txt", "w") as f:
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
    pr.specs("BEFORE ALL PROCESS")
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES")
    pr.fetch()
    pr.specs("AFTER FETCHING DATA")
    rf.model_creation()
    pr.specs("AFTER MODEL CREATION")
    rf.prediction()
    pr.specs("AFTER PREDICTION")
    rf.graph_plot()
    rf.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for random forest Modelling is :', elapsed_time, 'seconds')
  elif choice == 2:
    st_all=time.time()
    print("MLP MODELLING SELECTED")
    pr.combine_files()
    pr.fetch()
    multi.model_creation()
    multi.prediction()
    multi.graph_plot()
    multi.metrics() 
    et_all= time.time()
    elapsed_time = et_all - st_all 
    print('Complete Execution time for MLP is :', elapsed_time, 'seconds')
  elif choice == 3:
    st_all=time.time()
    print("LSTM MODELLING SELECTED")
    #pr.combine_files()
    pr.fetch()
    lstm.model_creation()
    lstm.prediction()
    lstm.graph_plot()
    lstm.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for LSTM is :', elapsed_time, 'seconds')
  elif choice== 4:
    st_all=time.time()
    print("ALL ML MODELLING SELECTED")
    pr.combine_files()  
    pr.fetch()
    rf.model_creation()
    multi.model_creation()
    lstm.model_creation()
    rf.prediction()
    multi.prediction()   
    lstm.prediction()
    rf.graph_plot()
    multi.graph_plot()
    lstm.graph_plot()
    rf.metrics()
    multi.metrics()
    lstm.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for ALL ML Modelling is :', elapsed_time, 'seconds')
  elif choice == 5:
    sys.exit() 
