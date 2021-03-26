# SDR4IoT ML

Deep Learning stuff with Tensorflow related to SDR4IoT

## /notebooks
doc.md : documentation of experiments and results   
fingerprinting_ble_2_classes.ipynb : notebook with first fingerprinting tests on scene 36-37 with one server   
fingerprinting_ble_3_classes_1_server_scene_31.ipynb : fingerprinting notebook with one server using scene 31 (3 classes)   
fingerprinting_ble_3_classes_1_server.ipynb : fingerprinting notebook with one server using scene 35-36-37 (3 classes)    
fingerprinting_ble_3_classes_3_servers.ipynb : fingerprinting notebook with three server using scene 35-36-37 (3 classes)    
localisation_ble_3classes_scene353637.ipynb : localisation with one server using scene 35-36-37 (regression)    
localisation_ble_classification.ipynb : first test localisation (do not use)   
localisation_ble_regression.ipynb : first test localisation (do not use)     

## /src 
data_prep.py : functions for input preparation    
dataset_sigmf_to_csv_ble.py : extract IQ data and metadata from sigmf files to csv files usable in the notebooks    
metrics_nn.py : functions to compute metrics and output graphs    
models_nn.py : various tuned models     
train_model.py : main, train a given model using a given dataset
tuner.py : various tuner for different models
tune_model.py : main, tune a given model using a given dataset with a given number of trials

## Makefile
make datasets : parses sigmf files to create csv datasets     
make train nb_server=(1 or 3) nn_type=(AlexNet, CNN2, ConvRNN, ResNet, CNNConv2D) exp_name=experiment name dataset_path= path to dataset (optional) : trains a given type of neural network, using a given dataset with a given number of servers      
make tune nb_server=(1 or 3) nn_type=(AlexNet, CNN2, ConvRNN, ResNet, CNNConv2D) exp_name=experiment name max_trial= max numb of trials dataset_path= path to dataset (optional)  : tunes a given type of neural network, using a given dataset with a given number of servers, during a given number of trials      
make notebook notebook_name= notebook_name   : executes a given notebook and saves it as a ipynb file and a html file     