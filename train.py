import pandas as pd
import logging
import argparse
import os.path as osp
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from ssrnet.model import SSRNet
from ssrnet.train.utils import makedirs, load_data_npz
from ssrnet.train.callbacks import DecayLearningRate
from ssrnet.train.generator import data_generator
from keras.utils.vis_utils import plot_model
from moviepy.editor import *


logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database npz file")
    parser.add_argument("--db", type=str, required=True,
                        help="database name")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="pretrained weights")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--netType1", type=int, required=True,
                        help="network type 1")
    parser.add_argument("--netType2", type=int, required=True,
                        help="network type 2")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    input_path = args.input
    db_name = args.db
    pretrained = args.pretrained
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    netType1 = args.netType1
    netType2 = args.netType2

    logging.debug("Loading data...")
    image, gender, age, image_size = load_data_npz(input_path)
    
    x_data = image
    y_data_a = age

    start_decay_epoch = [30,60]

    optMethod = Adam()

    stage_num = [3,3,3]
    lambda_local = 0.25*(netType1%5)
    lambda_d = 0.25*(netType2%5)

    model = SSRNet(image_size, stage_num, lambda_local, lambda_d)()
    save_name = f'ssrnet_{stage_num[0]}_{stage_num[1]}_{stage_num[2]}_{image_size}_{lambda_local}_{lambda_d}'
    model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})

    if pretrained is not None:
        model.load_weights(pretrained)
    
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    chkpt_path = osp.join(db_name, "checkpoints", save_name)
    model_path = osp.join(db_name, "models", save_name)
    log_path = osp.join(db_name, "logs", save_name)

    makedirs(model_path)
    makedirs(chkpt_path)
    makedirs(log_path)
    
    plot_model(model, to_file=osp.join(model_path, save_name+".png"))

    with open(osp.join(model_path, save_name+'.json'), "w") as f:
        f.write(model.to_json())
    
    decaylearningrate = DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(osp.join(chkpt_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate,
                tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
                ]

    logging.debug("Running training...")
    data_num = len(x_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    x_data = x_data[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    
    x_train = x_data[:train_num]
    x_test = x_data[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]


    hist = model.fit(x=data_generator(X=x_train, Y=y_train_a, batch_size=batch_size),
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(x_test, [y_test_a]),
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)

    logging.debug("Saving weights...")
    model.save_weights(osp.join(model_path, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(osp.join(model_path, 'history_'+save_name+'.h5'), "history")

if __name__ == '__main__':
    main()