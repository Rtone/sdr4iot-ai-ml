import logging
from data_prep import data_preparation_nn
from metrics_nn import plot_accuracy, test_metrics
from models_nn import build_model
import sys
import os
import pandas as pd
import logging
from tuner import launch_tuner
import sys
import os

#Main : tunes the selected model using the selected dataset, saves the best model and outputs its performances
#use python tune_model nb_server(1 or 3) nn_type experiment_name  max_trial  path to dataset(facultatif)

if __name__ == "__main__":
    experiment_name = sys.argv[3]
    os.mkdir(experiment_name)
    log_file = experiment_name + '/log_tune_model.log'
    logging.basicConfig(filename=log_file)
    nb_servers = sys.argv[1]
    max_trial = int(sys.argv[4])
    nn_type = sys.argv[2]
    if len(sys.argv) < 6:
        files = ('data/processed/scenario5_scene35.csv',
                 'data/processed/scenario5_scene36.csv',
                 'data/processed/scenario5_scene37.csv')
        idata = pd.concat([pd.read_csv(f) for f in files])
    else:
        idata = pd.read_csv(sys.argv[5])

    if nb_servers == '1':
        idata = idata[idata['Server_id'] == 11]

    idata = idata[idata['Len Packet'] == 1520]

    print('>> Launch Tuner for experiment {}, max trial {}, {} model'.format(
        experiment_name, max_trial, nn_type))
    launch_tuner(max_trial, idata, experiment_name, nn_type)
