# Uses 5 minutes/pixel (4320x2160) equirectengular dataset
# for the month of June
# Temperature and precipitation data from WorldClim
# Topography data from NOAA ETOPO
# Visual imagery from NASA Blue Marble

import numpy as np
import imagecodecs
import rasterio
from numpy import float32, ndarray

import keras
from keras import layers
from keras.layers import Input

temp_map = rasterio.open(r'temp_map.tif')
rain_map = rasterio.open(r'rain_map.tif')
elev_map = rasterio.open(r'elev_map.tif')
colorarray = imagecodecs.imread(r'colormap.png')

temp_array = np.array(temp_map.read())[0]
rain_array = np.array(rain_map.read())[0]
elev_array = np.array(elev_map.read())[0]

input_arrays = {
    'temp': temp_array,
    'rain': rain_array,
    'elev': elev_array
}

#split temp_map into 32x32 chunks
CHUNK_SIZE = 16
input_chunks = {key: np.zeros((1, CHUNK_SIZE, CHUNK_SIZE), dtype=float32) for key in ['temp', 'rain', 'elev']}
colorchunks = np.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 3), dtype=float32)

# for i in range(0, len(temp_array), 16):
#     for j in range(0, len(temp_array[i]), 16):

#reduced sample size for test purposes
for i in range(0, 128, CHUNK_SIZE):
    for j in range(0, 128, CHUNK_SIZE):
        
        #iterate through every key
        for key in input_arrays:
            input_chunks[key] = np.vstack((input_chunks[key], [input_arrays[key][i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]]))

        colorchunks = np.vstack((colorchunks, [colorarray[i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]]))
        print(f'{i} {j}')

def transform_input(chunks: ndarray, ceiling: int) -> ndarray:
    #flatten from (1, x, x, ...) to (1, x^2...)    
    chunks = chunks.reshape(chunks.shape[0], np.prod(chunks.shape[1:])) 
    #normalize to 0-1
    chunks = chunks/ceiling

    return chunks

# transform data for training
# input data
temp_chunks = transform_input(input_chunks['temp'], 40)
rain_chunks = transform_input(input_chunks['rain'], 3000)
elev_chunks = transform_input(input_chunks['elev'], 7000)
# target data
colorchunks = transform_input(colorchunks, 255)

#set up convolutional neural network model
def build_model():
    input_temp = Input(shape=(CHUNK_SIZE**2,), name="temp")
    input_rain = Input(shape=(CHUNK_SIZE**2,), name="rain")
    input_elev = Input(shape=(CHUNK_SIZE**2,), name="elev")

    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([input_temp, input_rain, input_elev])

    # Shrink output into 6 dimensions (rgb)(xy)
    output_color = layers.Dense(CHUNK_SIZE**2 * 3, activation='relu', name="color")(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs=[input_temp, input_rain, input_elev],
        outputs=[output_color],
    )

    #keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[keras.metrics.BinaryAccuracy(),
                        keras.metrics.FalseNegatives()])
    return model

model = build_model()
model.summary()
model.fit(
    {"temp": temp_chunks, "rain": rain_chunks, "elev": elev_chunks},
    {"color": colorchunks},
    epochs=2,
    batch_size=32,
)
