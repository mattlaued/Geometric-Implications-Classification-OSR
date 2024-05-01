import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from SupervisedAD_methods import *
from kdd import *

def main():
    df = get_df('data/KDDTrain+.txt', columns=columns, drop=False)
    df.head()

    test_df = get_df('data/KDDTest+.txt', columns=columns, drop=False)
    test_df.head()

    #  https://www.kaggle.com/code/avk256/nsl-kdd-anomaly-detection/notebook

    # lists to hold our attack classifications
    dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm',
                   'worm']
    probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                      'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop']

    # we will use these for plotting below
    attack_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']

    # map normal to 0, all attacks to 1
    is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
    test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

    # data_with_attack = df.join(is_attack, rsuffix='_flag')
    df['attack_flag'] = is_attack
    test_df['attack_flag'] = test_attack

    # map normal to 1, all attacks to 0
    is_normal = df.attack.map(lambda a: 1 if a == 'normal' else 0)
    test_normal = test_df.attack.map(lambda a: 1 if a == 'normal' else 0)

    df['normal_flag'] = is_normal
    test_df['normal_flag'] = test_normal


    # helper function to pass to data frame mapping
    def map_attack(attack):
        if attack in dos_attacks:
            # dos_attacks map to 1
            attack_type = 1
        elif attack in probe_attacks:
            # probe_attacks mapt to 2
            attack_type = 2
        elif attack in privilege_attacks:
            # privilege escalation attacks map to 3
            attack_type = 3
        elif attack in access_attacks:
            # remote access attacks map to 4
            attack_type = 4
        else:
            # normal maps to 0
            attack_type = 0

        return attack_type


    # map the data and join to the data set
    attack_map = df.attack.apply(map_attack)
    df['attack_map'] = attack_map

    test_attack_map = test_df.attack.apply(map_attack)
    test_df['attack_map'] = test_attack_map

    # categorical features
    features_to_encode = ['protocol_type', 'service', 'flag']

    # get numeric features, we won't worry about encoding these at this point
    # numeric_features = ['duration', 'src_bytes', 'dst_bytes']
    # Use all features
    numeric_features = list(set(df.columns[:-5]) - set(features_to_encode))


    def feat_eng(df, test_df, features_to_encode=features_to_encode, numeric_features=numeric_features):
        #     https://www.kaggle.com/code/avk256/nsl-kdd-anomaly-detection/notebook

        # get the intial set of encoded features and encode them
        encoded = pd.get_dummies(df[features_to_encode])
        test_encoded_base = pd.get_dummies(test_df[features_to_encode])

        # not all of the features are in the test set, so we need to account for diffs
        test_index = np.arange(len(test_df.index))
        column_diffs = list(set(encoded.columns.values) - set(test_encoded_base.columns.values))

        diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

        # we'll also need to reorder the columns to match, so let's get those
        column_order = encoded.columns.to_list()

        # append the new columns
        test_encoded_temp = test_encoded_base.join(diff_df)

        # reorder the columns
        test_final = test_encoded_temp[column_order].fillna(0)

        # model to fit/test
        to_fit = encoded.join(df[numeric_features])
        test_set = test_final.join(test_df[numeric_features])

        return to_fit, test_set


    data_train, data_test = feat_eng(df, test_df)
    data_train.head()

    scaler = StandardScaler()


    def get_x_y(df, data, classes=[0, 1]):
        indices = df['attack_map'].isin(classes)
        x = data[indices]
        y = df['attack_flag'][indices]

        return x.to_numpy(), y.to_numpy()

    x_train, y = get_x_y(df, data_train)
    X = scaler.fit_transform(x_train)

    np.random.seed(0)
    np.random.shuffle(X)
    np.random.seed(0)
    np.random.shuffle(y)

    x_testing, y_test = get_x_y(test_df, data_test, classes=test_classes)
    x_test = scaler.transform(x_testing)

    num_inputs = X.shape[-1]


    num_att_total = np.sum(y)
    num_normal = len(y) - num_att_total
    print("Baseline train AUPR: ", 1 - num_normal/len(y))

    num_att_total = np.sum(y_test)
    num_normal = len(y_test) - num_att_total
    print("Baseline overall AUPR: ", 1 - num_normal/len(y_test))

    val_counts = test_df['attack_map'].value_counts()[1:]

    for att in new_attacks:
        num_att = val_counts[att]
        print(f"Baseline AUPR {att}: ", num_att/(num_normal + num_att))


    # Train and Eval model

    separation = "RBF"
    bumped = ["", "b"]
    sigma = 0.5
    neurons = [num_inputs, num_inputs]
    hidden_layers = len(neurons)
    train = False  # train sigma. if NA, then False

    verbose = 1  # can change this to 0 to suppress verbosity during training
    plot = False
    shuffle = False
    val_split = 0.1
    repeats = 5
    epochs = 500
    batchnorm = False

    # lr = 3e-4

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss',
                                                      restore_best_weights=True)
    callbacks = [early_stopping]
    dropout = [0.0 for n in neurons]
    type_of_bump = 'gaussian'

    auprs = []

    for number_of_margins in range(1, 6):
        print("********************************************************************************")
        print("number_of_margins:", number_of_margins)

        results_df = test_model(
            X, y, x_test, y_test, num_inputs,
            anom_ids=new_attacks, anom_label=attack_labels, anom_label_data=test_df['attack_map'],
            separation=separation,
            bumped=bumped,
            sigma=sigma,
            train=train,  # train sigma. if NA, then False
            type_of_bump=type_of_bump, number_of_margins=number_of_margins,
            neurons=neurons,
            verbose=verbose,  # can change this to 0 to suppress verbosity during training
            plot=plot,
            shuffle=shuffle,
            val_split=val_split,
            repeats=repeats,
            epochs=epochs,
            batchnorm=batchnorm,
            lr=lr,
            callbacks=callbacks,
            save=False
        )
        aupr = tab_row(results_df)
        print("number_of_margins:", number_of_margins)
        print(aupr)
        auprs.append(f"{number_of_margins} & {aupr}")

    for aupr in auprs:
        print(aupr)


if __name__ == "__main__":
    main()