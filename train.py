# Uses 5 minutes/pixel (4320x2160) equirectengular dataset
# for the month of June
# Temperature and precipitation data from WorldClim
# Topography data from NOAA ETOPO
# Visual imagery from NASA Blue Marble

import numpy as np
import keras
from keras import layers
from preprocess import *

#set up convolutional neural network model
def build_model():
    input_temp = layers.Input(shape=(CHUNK_SIZE, CHUNK_SIZE, ), name="temp")
    input_elev = layers.Input(shape=(CHUNK_SIZE, CHUNK_SIZE, ),name="elev")
    input_rain = layers.Input(shape=(CHUNK_SIZE, CHUNK_SIZE, ),name="rain")

    # Merge all available features into a single large tensor via concatenation
    layer1 = layers.concatenate([input_temp, input_rain, input_elev], axis=-1)

    # Reshape into 1D tensor
    layer2 = layers.Flatten()(layer1)

    # Shrink output into 6 dimensions (rgb)(xy)layer2
    output_color = layers.Dense(CHUNK_SIZE**2 * 3, input_shape=(None, CHUNK_SIZE**2 * 3, ),  activation='sigmoid')(layer2)
    output_color = layers.Reshape((CHUNK_SIZE, CHUNK_SIZE, 3), name='color')(output_color)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs=[input_temp, input_rain, input_elev],
        outputs=[output_color],
    )

    model.compile(
        loss=keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
        optimizer=keras.optimizers.RMSprop(),
    )

    return model

import os

#input_arrays is a dictionary that holds the pre-processed maps 
# within input_arrays and colorarrays
# the 2D arrays within should be normalized to 0-1, but NOT flattened
input_arrays = \
{
    #todo - change MAP_NAMES constant, hardcoded values causing issues with ordering
    'temp': readfile(get_first_tif(r'training_dataset/temperature'), 0 , MAX_VALUES['temp']),
    'rain': readfile(get_first_tif(r'training_dataset/rainfall'), 0 , MAX_VALUES['rain']),
    'elev': readfile(get_first_tif(r'training_dataset/elevation'), 0 , MAX_VALUES['elev']),
}
colorarray = readfile(get_first_png(r'training_dataset/color'), 0 , 255)

#the map arrays should now be split into squares of size CHUNK_SIZE and flattened
temp_chunks, rain_chunks, elev_chunks, colorchunks = \
    [process_nan_values(chunks) for chunks in split_into_chunks(CHUNK_SIZE, list(input_arrays.values()) + [colorarray])]

#Confirmation for final data input shape and size
print(
    f'Non-empty chunks found (rain, temp, and elev should be equal): \n \
            Count\tShape\n\
    temp:\t{len(temp_chunks)}\t{temp_chunks.shape}\n\
    rain:\t{len(rain_chunks)}\t{rain_chunks.shape}\n\
    elev:\t{len(elev_chunks)}\t{elev_chunks.shape}\n\
    color:\t{len(colorchunks)}\t{colorchunks.shape}'
)
assert(temp_chunks.shape[1:] == (CHUNK_SIZE, CHUNK_SIZE, ))

#Model building and training
model = build_model()
model.summary()
model.fit(
    {"temp": temp_chunks, "rain": rain_chunks, "elev": elev_chunks},
    {"color": colorchunks},
    epochs=100,
)

model.save(r'model.keras')