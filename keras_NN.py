from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import os

from tensorflow.keras.mixed_precision import set_global_policy, Policy

# Set mixed precision policy
policy = Policy('mixed_float16')  # Use 'mixed_float16' for GPUs like NVIDIA A100
set_global_policy(policy)

img_width, img_height = 299, 299
IMG_SIZE = (img_width,img_height)
IMG_SHAPE = IMG_SIZE + (3,)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def create_generators(batch_size, seed):
    train_data_dir = "data/Train"
    validation_data_dir = "data/Valid"
    test_data_dir = "data/Test"

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed
    )
    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=seed
    )
    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=seed
    )
    return train_generator, validation_generator, test_generator


def run_experiment(batch_size, learning_rate, max_epoch, seed, dataset_size, patience, min_delta, 
                   verbose, factor, batch_norm=False, dropout=None, unfrozen_layers=None, load_model=None):
    
    train_generator, validation_generator, test_generator = create_generators(batch_size, seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = timestamp
    save_path_weights = f"data/weights/{timestamp}.h5"
    save_path_dataframe = f"data/results/{timestamp}.csv"
    
    inputs = keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    
    base_model = InceptionResNetV2(weights='imagenet',
                                   include_top=False,
                                   input_shape=IMG_SHAPE
                                  )
    
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1536))(x)     # Reshape for Conv2D (ensure 4D input)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    if batch_norm:
        x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Flatten()(x)
        
    output = keras.layers.Dense(7, activation='softmax')(x)
    
    start_time = time.time()

    model = keras.Model(inputs, output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()]
    )

    if load_model:
        model.load_weights(load_model)
    
    if unfrozen_layers:
        for layer in base_model.layers[-unfrozen_layers:]:
            layer.trainable = True

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    
    if factor:
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=factor,
            patience=patience,
            min_delta=1e-4,
            cooldown=2,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [reduce_lr, early_stopping,
                 ModelCheckpoint(filepath=save_path_weights, monitor='val_loss', mode = 'min', 
                                 save_best_only = True, save_weights_only=True)]
    else:
        callbacks = [early_stopping,
                 ModelCheckpoint(filepath=save_path_weights, monitor='val_loss', mode = 'min', 
                                 save_best_only = True, save_weights_only=True)]
    
    history = model.fit(train_generator, 
                        epochs=max_epoch, 
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        verbose=verbose)
    
    elapsed_time = round(-(start_time - time.time()) / 60, 1)
    epochs_trained = len(history.epoch)
    
    model.save_weights(save_path_weights)
    with open(f"{save_path_weights.split('.')[0]}.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    _, train_accuracy = model.evaluate(train_generator)
    _, valid_accuracy = model.evaluate(validation_generator)
    _, test_accuracy = model.evaluate(test_generator)

    print("epochs_trained", epochs_trained)

    df_res = pd.DataFrame({
        "batch_size": [batch_size],
        "learning_rate": [learning_rate],
        "max_epochs": [max_epoch],
        "patience": [patience],
        "min_delta": [min_delta],
        "factor": [factor],
        "epochs_trained": [epochs_trained],
        "batch_norm": [batch_norm],
        "dropout": dropout,
        "elapsed_time": [elapsed_time],
        "unfrozen_layers": [unfrozen_layers],
        "train_accuracy": [train_accuracy],
        "valid_accuracy": [valid_accuracy],
        "test_accuracy": [test_accuracy],
        "weights_path": [save_path_weights],
        })
    
    df_res.to_csv(save_path_dataframe, index=False)


def make_boxplots(results_df, experiment_number, train_data, variables_list, y_label):
    plt.figure(figsize=(18, 6))

    n_variables = len(variables_list)
    
    for variable_ind in range(n_variables):
        var_name = variables_list[variable_ind]
        x = results_df[var_name]
        
        plt.subplot(1, n_variables, variable_ind+1)
        sns.boxplot(data=results_df, x=y_label, y=var_name,
                   boxprops=dict(color='black'), medianprops=dict(color='white'))
        plt.title(f"Impact of {var_name} on {y_label}")
        plt.xlabel(y_label)
        plt.ylabel(var_name)
          
    plt.tight_layout()
    plt.savefig(f"plots/vgg16/{train_data}_experiment{experiment_number}_boxplot.png")
    plt.show()

def make_heatmap(results_df, variables_list, experiment_number, train_data, y_label, aggfunc="median"):
    # pivot table
    heatmap_data = results_df.pivot_table(
        values=y_label,
        index=variables_list[0],
        columns=variables_list[1],
        aggfunc=aggfunc
    )
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(heatmap_data, annot=True, cmap='blues', cbar_kws={'label': y_label})
    plt.title(f"Interaction of {variables_list[0]} and {variables_list[1]} on {y_label}")
    plt.xlabel(variables_list[1])
    plt.ylabel(variables_list[0])
    plt.savefig(f"plots/vgg16/{train_data}_experiment{experiment_number}_heatmap.png")
    plt.show()

import pandas as pd
import numpy as np
import os


def load_data(directory):
    dataframes = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            
            if "factor" not in df.columns:
                df["factor"] = [None] * df.shape[0]
            if "elapsed_time" not in df.columns:
                df["elapsed_time"] = [None] * df.shape[0]
                
            else:
                df["elapsed_time"] = df["elapsed_time"] / 60
            if "unfrozen_layers" not in df.columns:
                df["unfrozen_layers"] = [None] * df.shape[0]
                
            dataframes.append(df)
    
    final_df = pd.concat(dataframes, ignore_index=True)
    
    final_df["elapsed_time"] = final_df["elapsed_time"].abs()
    
    #final_df["learning_rate"][final_df[learning_rate] == ]
    
    final_df.rename(columns={"Droupout": 'Dropout'}, inplace=True)
    final_df["Dropout"] = final_df["Dropout"].fillna("None")
    final_df["factor"] = final_df["factor"].fillna("None")
    
    final_df["learning_rate"] = final_df["learning_rate"].astype("category")
    final_df["max_epochs"] = final_df["max_epochs"].astype("category")
    final_df["batch_size"] = final_df["batch_size"].astype("category")
    final_df["unfrozen_layers"] = final_df["unfrozen_layers"].astype("category")
    final_df["Batch Normalization"] = final_df["Batch Normalization"].astype("category")
    final_df["Batch Normalization Encoded"] = final_df["Batch Normalization"].astype("int")

    final_df.rename(columns={"Batch Normalization Encoded": "batch_norm_encoded", 
                             "Batch Normalization": "batch_norm", 
                             "Dropout": "dropout", 
                             "Base Model": "base_model", 
                             "Weights Path": "weights_path"}, inplace=True)
    
    return final_df

def load_model_and_predict(x_test, load_model_path, learning_rate, unfrozen_layers=None, batch_norm=False, dropout=None):
    IMG_SHAPE = (299, 299, 3)
    
    inputs = keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    
    base_model = InceptionResNetV2(weights='imagenet',
                                   include_top=False,
                                   input_shape=IMG_SHAPE
                                  )
    
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1536))(x)     # Reshape for Conv2D (ensure 4D input)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    if batch_norm:
        x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)  # Use global pooling instead of max pooling
    
    x = layers.Flatten()(x)  # Flatten before output
        
    output = keras.layers.Dense(7, activation='softmax')(x)
    
    start_time = time.time()

    #model = models.Model(inputs=base_model.input, outputs=output)
    model = keras.Model(inputs, output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()]
    )

    if load_model_path:
        model.load_weights(load_model_path)
    
    if unfrozen_layers:
        for layer in base_model.layers[-unfrozen_layers:]:
            layer.trainable = True
    
    preds = model.predict(x_test)
    y_pred = preds.argmax(axis=1)

    return preds, y_pred