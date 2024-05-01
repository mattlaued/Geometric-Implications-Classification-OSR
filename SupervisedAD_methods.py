import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss',
                                                  restore_best_weights=True)
strategy = tf.distribute.MirroredStrategy()


def viz_sigma(weight_callback):
    for k, v in weight_callback.weight_dict.items():
        # for i in range(len(v.shape)-1):
        #     v = v.mean(axis=-1)
        plt.plot(v, label=k)
        plt.title("Sigma against Epoch")
        plt.ylabel("Sigma")
        plt.xlabel("Epoch number")

    plt.legend()
    plt.show()


def get_metrics(y_pred, y_test, model_name, plot=True, pos_label=0, save=False):
    """

    :param y_pred: label predictions
    :param y_test: label ground truth
    :param model_name: name of model
    :param plot: whether to plot ROC and PR curves
    :param pos_label: attack label
    :param save: False to not save AUROC/AUPR plots, else path string to save
    :return: AUPR (Area under Precision-Recall Curve)
    """
    sign = 1

    # change pos_label from 1 to 0 since positive is attack
    if pos_label == 0:
        sign = -1
    fpr, tpr, threshold = metrics.roc_curve(y_test, sign * y_pred, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, threshold = metrics.precision_recall_curve(
        y_test, sign * y_pred, pos_label=pos_label)
    pr_auc = metrics.auc(recall, precision)

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=2)

        axs[0].plot(fpr, tpr, alpha=0.5,
                    label=f'AUROC = %0.5f' % roc_auc)
        axs[1].plot(recall, precision, alpha=0.5,
                    label=f'AUPR = %0.5f' % pr_auc)

        axs[0].set_title(f'ROC Curve: {model_name}')
        axs[0].legend(loc='lower right')
        axs[0].plot([0, 1], [0, 1], 'r--')
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 1])
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_xlabel('False Positive Rate')

        axs[1].set_title(f'Precision-Recall Curve: {model_name}')
        axs[1].legend(loc='lower right')
        axs[1].axhline(1 - y_test.sum() / len(y_test), linestyle='--', color='r')
        axs[1].set_xlim([0, 1])
        axs[1].set_ylim([0, 1])
        axs[1].set_ylabel('Precision')
        axs[1].set_xlabel('Recall')

        plt.tight_layout()
        if save:
            plt.savefig(save + "/Overall_AUPR_AUROC.png")
        plt.show()


    return pr_auc


def evaluate_predictions(y_pred, y_test, model_name="Model",
             plot=False, diff=False,
             indiv=False, anom_label=None, anom_label_data=None,
             pos_label=0, save=False):
    """

    :param y_pred: array of label predictions
    :param y_test: array of label ground truth
    :param model_name: name of model
    :param plot: bool whether to plot ROC and PR curves
    :param diff: bool whether to calculate avg diff btw pos and neg predictions
    :param indiv: list of anomaly ID describing individual anomaly types
    :param anom_label: list/dict of anomaly names (str) describing individual anomaly types
    :param anom_label_data: arr of anomaly ID of each sample in y_test
    :param pos_label: attack label
    :param save: False to not save AUROC/AUPR plots, else path string to save
    :return: AUPR (and indiv or diff results, if applicable)
    """

    # Evaluation
    aupr_test = get_metrics(y_pred, y_test, model_name, plot=plot, pos_label=pos_label, save=save)

    plt.title("Histogram for Predictions on Test Data")
    y_pred_normal = y_pred[y_test == (1 - pos_label)].squeeze()
    y_pred_anomalies = y_pred[y_test == pos_label].squeeze()
    plt.hist(y_pred_anomalies, bins=20, label="Anomalies", alpha=0.5)
    plt.hist(y_pred_normal, bins=20, label="Normal", alpha=0.5)
    plt.legend()
    if save:
        plt.savefig(f'{save}_{model_name}_test_pred.png')
    if plot:
        plt.show()
    else:
        plt.close()

    if indiv:
        # get indiv auprs for different attacks
        y_normal = len(y_pred_normal)

        aupr_attacks = []
        for i, anom in enumerate(indiv):
            y_anom = y_pred[anom_label_data == anom].squeeze()
            aupr_attack = get_metrics(
                    np.hstack((y_pred_normal, y_anom)),
                    np.hstack((np.ones(y_normal), np.zeros(len(y_anom)))),
                    model_name, plot=plot)
            aupr_attacks.append(aupr_attack)
        fig, ax = plt.subplots(nrows=1, ncols=len(indiv), figsize=(3.5 * len(indiv), 7))
        for i, anom in enumerate(indiv):
            y_anom = y_pred[anom_label_data == anom].squeeze()

            ax[i].set_title(f"Anomaly Predictions")
            ax[i].hist(y_anom, bins=20, label=f"Anomaly {anom}: {anom_label[anom]}", alpha=0.5)
            ax[i].hist(y_pred_normal, bins=20, label="Normal", alpha=0.5)
            ax[i].text(0.0, 0.9, f'AUPR: {aupr_attacks[i]:.5}', style='italic', horizontalalignment='left',
                verticalalignment='center', transform=ax[i].transAxes)
                    # bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            ax[i].legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"{save}_FineGrained_AUPR_AUROC.png")
        if plot:
            plt.show()
        else:
            plt.close()

        try:
            display(pd.DataFrame(data={model_name: aupr_attacks}))
        except:
            print(pd.DataFrame(data={model_name: aupr_attacks}))


        return aupr_test, aupr_attacks

    if diff:
        y_pos = np.mean(y_pred_normal)
        y_neg = np.mean(y_pred_anomalies)
        diff_mean = y_pos - y_neg
        print(f"Average Difference between Positive and Negative Class: {diff_mean}")

        return aupr_test, diff_mean

    return aupr_test


# Shallow Model Evaluation

def eval_plot(y_test, y_pred, model_name, plot=True):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred, pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, threshold = metrics.precision_recall_curve(
        y_test, y_pred, pos_label=0)
    pr_auc = metrics.auc(recall, precision)

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(9, 4.5)

        axs[0].set_title('ROC Curve')
        axs[0].plot(fpr, tpr, 'purple', alpha=0.8,
                    label=f'{model_name} AUC = %0.2f' % roc_auc)
        axs[0].legend(loc='lower right')
        axs[0].plot([0, 1], [0, 1], 'r--')
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 1])
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_xlabel('False Positive Rate')

        axs[1].set_title('Precision-Recall Curve')
        axs[1].plot(recall, precision, 'purple', alpha=0.8,
                    label=f'{model_name} AUC = %0.2f' % pr_auc)
        axs[1].legend(loc='lower right')
        axs[1].axhline(y_test.sum() / len(y_test), linestyle='--', color='r')
        axs[1].set_xlim([0, 1])
        axs[1].set_ylim([0, 1])
        axs[1].set_ylabel('Precision')
        axs[1].set_xlabel('Recall')

        plt.tight_layout()
        plt.show()

    return pr_auc

# Bump Activation

def bump_activation(x, sigma=0.5, type_of_bump='gaussian', number_of_margins: int = 1, dist=None):
    """
    bump activation
    :param x: input
    :param sigma: variance / width parameter
    :param type_of_bump: 'gaussian' or 'tanh'
    :param number_of_margins: number of separate margins, int
    :param dist: distance btw bumps
    :return:
    """

    if dist is None:
        dist = sigma

    if number_of_margins == 1:
        if type_of_bump == 'gaussian':
            return tf.math.exp(-0.5 * tf.math.square(x/sigma))
        return tf.math.tanh(tf.math.square(sigma / x))
    s = sum([
        bump_activation(x - i * dist, sigma, number_of_margins=1) for i in range(number_of_margins)
    ])
    # normalise so max is 1
    middle = number_of_margins // 2
    normalise = sum([
        bump_activation((middle - i) * dist, sigma, number_of_margins=1) for i in range(number_of_margins)
    ])
    return s / normalise


# Deep Model Training

class Bump(tf.keras.layers.Layer):

    def __init__(self, sigma=0.5, trainable=True, name="bump",
                 type_of_bump='gaussian', number_of_margins: int = 1, **kwargs):
        super(Bump, self).__init__(**kwargs)
        self.supports_masking = True
        self.sigma = sigma
        self.trainable = trainable
        self._name = name
        self.type_of_bump = type_of_bump
        self.number_of_margins = number_of_margins

    def build(self, input_shape):
        self.sigma_factor = K.variable(self.sigma,
                                       dtype=K.floatx(),
                                       name='sigma_factor')
        if self.trainable:
            self._trainable_weights.append(self.sigma_factor)

        super(Bump, self).build(input_shape)

    def call(self, inputs, mask=None):
        return bump_activation(inputs, self.sigma_factor,
                               type_of_bump=self.type_of_bump, number_of_margins=self.number_of_margins)

    def get_config(self):
        config = {'sigma': self.get_weights()[0] if self.trainable else self.sigma,
                  'trainable': self.trainable}
        base_config = super(Bump, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma, initializer, dim=1, name="bump", mu_param=True, beta_param=False,
                 type_of_bump='gaussian', number_of_margins: int = 1, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)
        self.initializer = initializer
        self._name = name
        self.mu_param = mu_param
        self.beta_param = beta_param
        self.type_of_bump = type_of_bump
        self.number_of_margins = number_of_margins

        # dim is 1D (tabular) or 2D (sequence)
        self.dim = dim

    def build(self, input_shape):
        if self.mu_param:
            self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[-1]), self.units),
                                  initializer=self.initializer,
                                  trainable=True)
        else:
            self.mu = tf.ones((int(input_shape[-1]), self.units))
        if self.beta_param:
            self.beta = self.add_weight(name='beta',
                                        shape=(1,),
                                        initializer=self.initializer,
                                        trainable=True)
        else:
            self.beta = 1.
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        # calculate square l2 norm
        if self.dim == 1:
            diff = K.expand_dims(inputs) - self.mu
            l2 = K.sum(K.pow(diff / self.beta, 2), axis=1)
        else:
            x = inputs[:, :, :, None]
            mu = self.mu[None, None, :, :]
            diff = tf.subtract(x, mu)
            l2 = K.sum(K.pow(diff / self.beta, 2), axis=-2)
        res = bump_activation(l2, self.gamma,
                              type_of_bump=self.type_of_bump, number_of_margins=self.number_of_margins)
        return res

    def compute_output_shape(self, input_shape):
        if self.dim == 1:
            return (input_shape[0], self.units)
        return input_shape[:-1] + (self.units,)


class GetWeights(tf.keras.callbacks.Callback):

    def __init__(self, layer_names=["bump"]):
        super(GetWeights, self).__init__()
        self.weight_dict = {}
        self.layer_names = layer_names
        for layer in layer_names:
            self.weight_dict[layer] = []

    def on_epoch_end(self, epoch, logs=None):
        if len(self.layer_names) > 0:
            for layer in self.layer_names:
                self.weight_dict[layer].append(self.model.get_layer(layer).get_weights()[0])


# Build Models
def build_layer(activation, input_layer, sigma=0.5, train=False,
                type_of_bump='gaussian', number_of_margins=1, layer_number=1,
                seed=0, neurons=5, batchnorm=False, regulariser=None):
    initialiser = tf.keras.initializers.GlorotUniform(seed=seed)

    if activation == "r":
        if layer_number == "last":
            mu_param = False
        else:
            mu_param = True
        layer = RBFLayer(neurons, gamma=1.0, initializer=initialiser, name=f"bump{str(layer_number)}",
                         mu_param=mu_param, beta_param=train)(input_layer)

        if batchnorm:
            layer = tf.keras.layers.BatchNormalization()(layer)

    else:
        hidden = tf.keras.layers.Dense(neurons,
                                       kernel_initializer=initialiser, kernel_regularizer=regulariser)(input_layer)

        if batchnorm:
            hidden = tf.keras.layers.BatchNormalization()(hidden)

        if activation == "b":
            layer = Bump(sigma=sigma, trainable=train,
                         name=f"bump{str(layer_number)}",
                         type_of_bump=type_of_bump, number_of_margins=number_of_margins)(hidden)
        elif activation == "s":
            layer = tf.math.sigmoid(hidden)
        else:
            layer = tf.nn.leaky_relu(hidden, alpha=0.01)

    return layer


def create_model(separation, activation, hidden_layers, num_inputs,
                 hidden_neurons=[40, 20, 10, 5], dropout=[0.0, 0.0, 0.0, 0.0], lr=0.001,
                 regularisation=[None, None, None, None],
                 sigma=0.5, train=False, type_of_bump='gaussian', number_of_margins=1,
                 loss='binary_crossentropy', batchnorm=False,
                 seed=0, name_suffix=""):
    sep = {"RBF": "r", "ES": "b", "HS": "s"}

    tf.keras.utils.set_random_seed(seed)

    input_layer = tf.keras.Input(shape=(num_inputs,))

    if type(hidden_neurons) is list:

        if len(hidden_neurons) == 0:
            out = build_layer(sep[separation], input_layer, sigma=sigma,
                              type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                              layer_number="last", seed=seed + 2023,
                              neurons=1, batchnorm=False)

        else:

            hidden_layers = len(hidden_neurons)
            if isinstance(activation, str):
                activation = [activation for _ in range(len(hidden_layers))]

            hidden = input_layer

            for i, n in enumerate(hidden_neurons):
                hidden = build_layer(activation[i], hidden, sigma=sigma, train=train,
                                     type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                                     layer_number=1 + i, seed=seed + 42 * i, neurons=n,
                                     batchnorm=batchnorm, regulariser=regularisation[i])
                if dropout[i] > 0.:
                    hidden = tf.keras.layers.Dropout(dropout[i])(hidden)

            out = build_layer(sep[separation], hidden, sigma=sigma,
                              type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                              layer_number="last", seed=seed + 2023, neurons=1, batchnorm=False)

    else:
        if isinstance(activation, str):
            activation = [activation for _ in range(len(hidden_layers))]
        hidden1 = build_layer(activation[0], input_layer, sigma=sigma, train=train,
                              type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                              layer_number=1, seed=seed + 42)
        hidden2 = build_layer(activation[1], hidden1, sigma=sigma, train=train,
                              type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                              layer_number=2, seed=seed + 123)

        if hidden_layers == 2:

            out = build_layer(sep[separation], hidden2, sigma=sigma,
                              type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                              layer_number="last", seed=seed + 2023, neurons=1, batchnorm=False)

        elif hidden_layers == 3:

            hidden3 = build_layer(activation[2], hidden2, sigma=sigma, train=train,
                                  type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                                  layer_number=3, seed=seed + 1234)
            out = build_layer(sep[separation], hidden3, sigma=sigma,
                              type_of_bump=type_of_bump, number_of_margins=number_of_margins,
                              layer_number="last", seed=seed + 2023, neurons=1, batchnorm=False)

    activation_list = "".join(["l" if a == "" else a for a in activation])
    model = tf.keras.Model(inputs=input_layer, outputs=out,
                           name=f"{separation}{hidden_layers}{activation_list}{name_suffix}")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=loss)

    return model


# Train and Evaluate Models

def train_eval(model, X, y, x_test, y_test, epochs=1000, train=False, hidden_layers=2,
               verbose=0, shuffle=False, plot=False,
               val_split=0.1, callbacks=[early_stopping], seed=0, diff=False, save=False,
               indiv=None, anom_label=None, anom_label_data=None, pos_label=0):
    """

    :param model:
    :param X:
    :param y: training label -- 0 for anom, 1 for normal
    :param x_test:
    :param y_test:
    :param epochs:
    :param train:
    :param hidden_layers:
    :param verbose:
    :param shuffle:
    :param plot:
    :param val_split:
    :param callbacks:
    :param seed:
    :param diff:
    :param save:
    :param indiv:
    :param anom_label:
    :param anom_label_data:
    :param pos_label:
    :return:
    """
    # Train the model
    tf.keras.utils.set_random_seed(seed)
    cbs = callbacks.copy()
    if train:
        # learnable sigma
        get_weights = GetWeights(layer_names=[f"bump{i}" for i in range(1, hidden_layers + 1)])
        cbs.append(get_weights)
        model.fit(X, y, epochs=epochs, verbose=verbose, shuffle=shuffle,
                  validation_split=val_split, callbacks=cbs)
        # viz_sigma(get_weights)

    else:
        model.fit(X, y, epochs=epochs, verbose=verbose, shuffle=shuffle,
                  validation_split=val_split, callbacks=cbs)

    y_train = model.predict(X)
    aupr_train = get_metrics(y_train, y, model.name, plot=plot, pos_label=pos_label, save=save)

    y_pred = model.predict(x_test)
    aupr_test = evaluate_predictions(y_pred, y_test, model_name="Model",
                                     plot=False, diff=False,
                                     indiv=indiv, anom_label=anom_label, anom_label_data=anom_label_data,
                                     pos_label=0, save=save)

    if indiv or diff:
        return aupr_train, aupr_test[0], aupr_test[1]

    return aupr_train, aupr_test


def test_model(
        X_train, y_train, X_test, y_test, num_inputs,
        anom_ids=None, anom_label=None, anom_label_data=None,
        separation="ES",
        bumped="b",
        sigma=3.,
        train=False,  # train sigma. if NA, then False
        type_of_bump='gaussian', number_of_margins=1,
        neurons=[],
        verbose=1,  # can change this to 0 to suppress verbosity during training
        plot=False,
        shuffle=False,
        val_split=0.0,
        repeats=3,
        epochs=500,
        batchnorm=True,
        lr=3e-4,
        callbacks=[],
        save=False
):
    hidden_layers = len(neurons)
    dropout = [0.0 for n in neurons]

    auprs_train = []
    auprs_test = []
    aupr_attacks = dict()

    # models_hs = []

    # Train and Evaluate the Model
    for i in range(repeats):

        with strategy.scope():
            # Create the model
            tf.keras.utils.set_random_seed(i)

            model = create_model(separation, activation=bumped, hidden_layers=hidden_layers, num_inputs=num_inputs,
                                 hidden_neurons=neurons, batchnorm=batchnorm, dropout=dropout,
                                 sigma=sigma, train=train, type_of_bump=type_of_bump,
                                 number_of_margins=number_of_margins, loss='binary_crossentropy', lr=lr,
                                 seed=i)

            if i == 0:
                model.summary()
            # Train the model
            aupr_train, aupr_test, aupr_attack = train_eval(model, X_train, 1 - y_train, X_test, 1 - y_test,
                                                            epochs=epochs, hidden_layers=hidden_layers,
                                                            train=train, verbose=verbose, shuffle=shuffle, plot=plot,
                                                            val_split=val_split, callbacks=callbacks, seed=i,
                                                            indiv=anom_ids, anom_label=anom_label,
                                                            anom_label_data=anom_label_data, save=save)

        # models_hs.append(model)

        print(f"AUPR Train Run {i + 1}: {aupr_train}")
        print(f"AUPR Test Run {i + 1}: {aupr_test}")
        print(f"AUPR Indiv Test Run {i + 1}: {aupr_attack}")
        auprs_train.append(aupr_train)
        auprs_test.append(aupr_test)
        aupr_attacks[i] = aupr_attack

    print(f"AUPR (Train): {np.mean(auprs_train)}+-{np.std(auprs_train)}")
    print(f"AUPR (Test): {np.mean(auprs_test)}+-{np.std(auprs_test)}")

    # last row is overall aupr
    results_df = pd.DataFrame(data=aupr_attacks, index=anom_ids)
    results_df.loc[anom_ids[-1] + 1] = auprs_test
    results_df['AUPR Mean'] = results_df.mean(axis=1)
    results_df['AUPR Std'] = results_df.iloc[:, :-1].std(axis=1)


    print(results_df)
    # print(f"Average Distance between Means: {np.mean(diff_means)}+-{np.std(diff_means)}")
    return results_df


def tab_row(df):
    s = ""
    means = df['AUPR Mean']
    stds = df['AUPR Std']
    for mean, std in zip(means, stds):
        mean = round(mean, 3)
        std = round(std, 3)
        s += f"{mean}$\\pm${std} & "
    return s[:-3]
