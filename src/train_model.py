import logging
from data_prep import data_preparation_nn
from metrics_nn import plot_accuracy, test_metrics
from models_nn import build_model
import sys
import os
import pandas as pd

#Main : trains the selected model using the selected dataset, saves the trained model and outputs its performances
#use python train_model.py nb_server(1 or 3)   type of nn (AlexNet, CNN2, ConvRNN, ResNet, CNNConv2D) experiment_name  path to dataset(facultatif)

len_shift_dict = {
    'AlexNet': [450, 30],
    'CNN2': [200, 50],
    'ConvRNN': [200, 50],
    'ResNet': [200, 50],
    'CNNConv2D': [200, 50]
}

if __name__ == "__main__":
    experiment_name = sys.argv[3]
    os.mkdir(experiment_name)
    log_file = experiment_name + '/log_train_model.log'
    logging.basicConfig(filename=log_file)
    nb_servers = sys.argv[1]
    nn_type = sys.argv[2]

    if len(sys.argv) < 5:
        files = ('data/processed/scenario5_scene35.csv',
                 'data/processed/scenario5_scene36.csv',
                 'data/processed/scenario5_scene37.csv')
        idata = pd.concat([pd.read_csv(f) for f in files])
    else:
        idata = pd.read_csv(sys.argv[4])

    if nb_servers == '1':
        idata = idata[idata['Server_id'] == 11]

    len_slice = len_shift_dict[nn_type][0]
    shift = len_shift_dict[nn_type][1]

    idata = idata[idata['Len Packet'] == 1520]
    print('>> Launch Trainer for experiment {}, {} model'.format(
        experiment_name, nn_type))
    X_train, X_test, y_train, y_test, input_shape, batch_size = data_preparation_nn(
        idata, len_slice, shift)
    model, callback = build_model(nn_type, input_shape, experiment_name)
    history = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=300,
                        verbose=1,
                        validation_split=0.2,
                        callbacks=[callback])
    model_file = experiment_name + '/' + nn_type
    model.save(model_file)
    plot_accuracy(history, experiment_name)
    test_metrics(model, y_test, X_test, experiment_name)
