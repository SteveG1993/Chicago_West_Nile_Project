import numpy as np
import pandas as pd
import time
import os
import sys
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class P4_Data_Class():
    def __init__(self):
        print('Initialized')
    def add_two_numbers(self,x=0,y=0):
        return x + y


def write_params(df):
    suffix = '.csv'
    timestr = time.strftime("%Y%m%d_%H%M%S")
    base_filename = 'features_' + timestr
    #os.path.join(dir_name, base_filename + suffix)
    file_name = os.path.join(base_filename + suffix)
    initial_column_list = list(df.columns.values)
    f = open(file_name, "w+")
    f.write("\n".join(map(lambda x: str(x), initial_column_list)))
    f.close()
    return file_name

def read_params(feature_file_name):
    print('Inside funct file name',feature_file_name)
    feature_text_file = open(feature_file_name, 'r')
    yourResult = [line.split(',') for line in feature_text_file.readlines()]
    df_drop_cols = pd.DataFrame(yourResult,columns=['col_name','keep'])
    df_drop_cols = df_drop_cols[df_drop_cols['keep']=='No\n']
    retval = drop_cols = df_drop_cols['col_name'].tolist()
    return retval


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.rcParams.update({'font.size': 16})
    #plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')


def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()



 def dataframe_differences(first_df,second_df,update_first=False,update_second=False):
    only_in_first = set(first_df.columns) .difference(set(second_df.columns))
    only_in_second = set(second_df.columns).difference(set(first_df.columns))
    print ('Only in first BEFORE:' ,set(first_df.columns) .difference(set(second_df.columns)))
    print ('Only in second BEFORE:',set(second_df.columns).difference(set(first_df.columns)))
    if update_first == True:
        for s in only_in_second:
            first_df[s]=0
    if update_second == True:
        for f in only_in_first:
            second_df[f]=0
    if update_first or update_second:
        print('--------------------------------------------')
        print ('Only in first AFTER:' ,set(first_df.columns) .difference(set(second_df.columns)))
        print ('Only in second AFTER:',set(second_df.columns).difference(set(first_df.columns)))
