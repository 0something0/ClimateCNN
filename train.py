# Uses 5 minutes/pixel (4320x2160) equirectengular dataset
# for the month of June
# Temperature and precipitation data from WorldClim
# Topography data from NOAA ETOPO
# Visual imagery from NASA Blue Marble

import numpy as np

from numpy import NaN, float32, ndarray

import keras
from keras import layers
from preprocess import *

#set up convolutional neural network model
def build_model():
    input_temp = layers.Input(shape=(1, CHUNK_SIZE**2, ), name="temp")
    input_elev = layers.Input(shape=(1, CHUNK_SIZE**2, ),name="elev")
    input_rain = layers.Input(shape=(1, CHUNK_SIZE**2, ),name="rain")

    # Merge all available features into a single large vector via concatenation
    concated = layers.concatenate([input_temp, input_rain, input_elev], axis=-1)

    # Shrink output into 6 dimensions (rgb)(xy)
    output_color = layers.Dense(CHUNK_SIZE**2 * 3, input_shape=(None, CHUNK_SIZE**2 * 3) ,  activation='sigmoid', name="color")(concated)

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


#input_arrays = {key: readfile(rf'{key}.tif', 0 , MAX_VALUES[key]) for key in MAP_NAMES}

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
    [flatten_input(chunks) for chunks in split_into_chunks(CHUNK_SIZE, list(input_arrays.values()) + [colorarray])]

print(f'{len(temp_chunks)} {len(rain_chunks)} {len(elev_chunks)} {len(colorchunks)}')

#add a third dimension to each variable, for training purposes
temp_chunks, rain_chunks, elev_chunks, colorchunks = \
    [np.expand_dims(chunk, axis=1) for chunk in [temp_chunks, rain_chunks, elev_chunks, colorchunks]]

assert(temp_chunks.shape[1:] == (1, 256))
print(temp_chunks.shape)

model = build_model()
model.summary()

model.fit(
    {"temp": temp_chunks, "rain": rain_chunks, "elev": elev_chunks},
    {"color": colorchunks},
    epochs=100,
    #batch_size=43,
)

#save model to model.keras
model.save(r'model.keras')