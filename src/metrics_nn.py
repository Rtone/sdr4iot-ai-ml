import seaborn as sns
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from matplotlib import pyplot
import logging
import sys

#Computes metrics to evaluate a given model performance and print/save them


#plot the accuracy and the val_accuracy vs epochs during learning, saves the graph
def plot_accuracy(history, experiment_name):
    i = np.arange(len(history.history['loss']))
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.plot(i, history.history['accuracy'], label='accuracy')
    plt.plot(i, history.history['val_accuracy'], label='val_accuaracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3,
               ncol=2,
               mode="expand",
               borderaxespad=0.)
    plt.show()
    fig_file = experiment_name + '/acc_vs_epoch.png'
    plt.savefig(fig_file)


#For classification : computes various metrics on the test set: confusion matrix, f1-score (global and for each class), global accuracy
def test_metrics(model, y_test, X_test, experiment_name):
    predictions = model.predict(X_test)
    predictions = [np.argmax(y, axis=0, out=None) for y in predictions]
    y_test_cat = [np.argmax(y, axis=0, out=None) for y in y_test]

    present_class_list = np.unique(y_test_cat)

    old_stdout = sys.stdout
    log_file = experiment_name + '/log_train_model.log'
    log_file = open(log_file, "a")
    sys.stdout = log_file

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test_cat,
                                predictions,
                                labels=present_class_list)

    #F1-score
    print("f1_score for each class")
    f1_each = f1_score(y_test_cat,
                       predictions,
                       average=None,
                       labels=present_class_list)
    print(f1_each)

    #F1-score
    print("Global f1_score")
    print(f1_score(y_test_cat, predictions, average='weighted'))

    #accyracy
    print("Global accuracy")
    print(accuracy_score(y_test_cat, predictions))

    sys.stdout = old_stdout
    log_file.close()

    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    plt.figure()
    sns.heatmap(conf_mat,
                cmap=cmap,
                xticklabels=present_class_list,
                yticklabels=present_class_list)
    mat_file = experiment_name + '/conf_mat.png'
    plt.savefig(mat_file)
