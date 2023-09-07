import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l1
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import nltk
from nltk.corpus import stopwords


def plot_graphs(history: object, metric: str) -> None:
    '''
    Plot graphs from tf history

    Parameters
    ----------
    history : obj
        tf history
    metric : str
        metric
    
    Returns
    -------
    None
    '''
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def remove_stop_words(text: str) -> str:
    '''
    Remove stop words from text

    Parameters
    ----------
    text : str
        text to be filtered
    
    Returns
    -------
    filtered string
    '''
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


if __name__ == '__main__':

    df_tweets = pd.read_csv('tweets.csv')
    x = df_tweets['Text']
    y = df_tweets['CB_Label']

    x_filt = x.apply(remove_stop_words)

    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True)

    lst_cv_acc = []
    lst_cv_loss = []

    for i, (train_ind, test_ind) in enumerate(kf.split(x_filt)):

        x_train, x_test = x_filt[train_ind], x_filt[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        encoder = tf.keras.layers.TextVectorization()
        encoder.adapt(x_train.values)

        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
                metrics=['accuracy'])

        history = model.fit(x=x_train, y=y_train, epochs=10,
                        validation_split=.2)

        test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
        lst_cv_acc.append(test_acc)
        lst_cv_loss.append(test_loss)

        print('K-fold:', i)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plot_graphs(history, 'accuracy')
        plt.subplot(1, 2, 2)
        plot_graphs(history, 'loss')
        plt.show()

    print('K-Fold Acc:', lst_cv_acc)
    print('K-Fold Loss:', lst_cv_loss)
