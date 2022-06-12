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

temp_chunks, rain_chunks, elev_chunks, colorchunks = [get_first_tif(file_path) for file_path in [r'training_dataset/temperature', r'training_dataset/rainfall', r'training_dataset/elevation']] + [get_first_png( r'training_dataset/color')]
temp_chunks, rain_chunks, elev_chunks, colorchunks = read_multi_files([temp_chunks, rain_chunks, elev_chunks, colorchunks], [40, 3000, 7000, 255])
temp_chunks, rain_chunks, elev_chunks, colorchunks = [chunks for chunks in split_into_chunks(CHUNK_SIZE, [temp_chunks, rain_chunks, elev_chunks, colorchunks])]
temp_chunks, rain_chunks, elev_chunks, colorchunks = [process_nan_values(chunks) for chunks in [temp_chunks, rain_chunks, elev_chunks, colorchunks]]

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