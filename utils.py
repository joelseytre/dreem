import pandas as pd
import numpy as np
import os
import local_info


def load_activations(is_FFT):
    # loads the activations (of the NN) that were stored (storing => don't need to train the NN each time)
    print("Reading activations dataset...")
    if is_FFT:
        activations_train = np.array(pd.read_csv(local_info.data_path + 'activations_train_fft.csv').values)
        activations_valid = np.array(pd.read_csv(local_info.data_path + 'activations_valid_fft.csv').values)
        activations_test = np.array(pd.read_csv(local_info.data_path + 'activations_test_fft.csv').values)
        print("Activations (FFT) have been read from csv! (train shape is %s)" % str(np.shape(activations_train)))
    else:
        activations_train = np.array(pd.read_csv(local_info.data_path + 'activations_train.csv').values)
        activations_valid = np.array(pd.read_csv(local_info.data_path + 'activations_valid.csv').values)
        activations_test = np.array(pd.read_csv(local_info.data_path + 'activations_test.csv').values)
        # activations_train = np.loadtxt(local_info.data_path + 'activations_train.csv', delimiter=",")
        # activations_valid = np.loadtxt(local_info.data_path + 'activations_valid.csv', delimiter=",")
        # activations_test = np.loadtxt(local_info.data_path + 'activations_test.csv', delimiter=",")
        print("Activations have been read from csv! (train shape is %s)" % str(np.shape(activations_train)))
    return activations_train, activations_valid, activations_test


def store_activations(activ_train, activ_valid, activ_test, is_FFT):
    # stores the activations (of the NN) for future usage (storing => don't need to train the NN each time)
    os.chdir(local_info.data_path)
    if is_FFT:
        pd.DataFrame(activ_train).to_csv('activations_train_FFT.csv', encoding='utf-8', index=False)
        pd.DataFrame(activ_valid).to_csv('activations_valid_FFT.csv', encoding='utf-8', index=False)
        pd.DataFrame(activ_test).to_csv('activations_test_FFT.csv', encoding='utf-8', index=False)
        print("Activations (FFT) have been stored! (train shape is %s)\n" % str(np.shape(activ_train)))
    else:
        pd.DataFrame(activ_train).to_csv('activations_train.csv', encoding='utf-8', index=False)
        pd.DataFrame(activ_valid).to_csv('activations_valid.csv', encoding='utf-8', index=False)
        pd.DataFrame(activ_test).to_csv('activations_test.csv', encoding='utf-8', index=False)
        print("Activations have been stored! (train shape is %s)\n" % str(np.shape(activ_train)))


def load_dataset(small_dataset, storing_small_dataset):
    # loads the dataset from csv (small dataset for debugging
    print("Reading raw dataset...")
    if small_dataset:
        print("Small dataset chosen")
        df_train = pd.read_csv(local_info.data_path + 'extract_train.csv')
        df_test = pd.read_csv(local_info.data_path + 'extract_test.csv')
    else:
        print("Full dataset chosen")
        df_train = pd.read_csv(local_info.data_path + 'train.csv')
        df_test = pd.read_csv(local_info.data_path + 'test.csv')
    print("Done reading dataset! \n")

    # stores an extract of 500 rows instead of 50k for debugging purposes
    if storing_small_dataset:
        store_small_dataset(df_train, 500, "train")
        store_small_dataset(df_test, 500, "test")

    return df_train, df_test



def normalize_dataframe(df, mean, var):
    # normalizes the entire df by substracting mean and dividing by var
    df_out = df.sub(mean, axis=0)
    df_out = df_out.div(var, axis=0)
    return df_out


def store_small_dataset(df, length, name):
    # create small dataset for debugging purposes
    print("Storing extract of " + name + " data...")
    os.chdir(local_info.data_path)
    temp_df = df[0:length]
    temp_df.to_csv('extract_' + name + '.csv', encoding='utf-8', index=False)
    print("Done storing %s extract at location %s ! \n" % (name, os.getcwd()))
