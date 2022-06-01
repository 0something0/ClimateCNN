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

CHUNK_SIZE = 16
MAP_NAMES = ['temp', 'rain', 'elev']
MAP_BOUNDS = {'temp':40, 'rain':3000, 'elev':7000}

#Reads from f'{filename}_map.tif', clamps to min and max, and removes NaNs'
#seems to be lower this way, better compare between processing input before and after splitting
def readfile(filename: str, min: int, max: int):
    
    if filename.find('.tif') != -1:
        arr =  np.array(rasterio.open(filename).read())[0]
    elif filename.find('.png') != -1:
        arr = imagecodecs.imread(filename)

    arr = np.nan_to_num(arr)
    arr = np.clip(arr, min, max)
    arr = (arr - min)/max
    return arr

input_arrays = {key: readfile(rf'{key}_map.tif', 0 , MAP_BOUNDS[key]) for key in MAP_NAMES}
colorarray = readfile(r'colormap.png', 0 ,255)

#split temp_map into chunks
input_chunks = {key: np.zeros((1, CHUNK_SIZE, CHUNK_SIZE), dtype=float32) for key in MAP_NAMES}

colorchunks = np.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 3), dtype=float32)

#reduced sample size for test purposes, should be len(input_arrays['temp']) and len(input_arrays['temp'][i]
row_start = int(len(input_arrays['temp']) / 2)
col_start = int(len(input_arrays['temp'][1]) / 2 + 512)

for i in range(row_start, + 128, CHUNK_SIZE):
    for j in range(col_start, col_start + 128, CHUNK_SIZE):

        #check if temp chunk is empty, go to next iteration
        if np.sum(input_arrays['temp'][i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]) == 0:
            continue

        #iterate through every key
        for key in input_arrays:
            input_chunks[key] = np.vstack((input_chunks[key], [input_arrays[key][i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]]))

        colorchunks = np.vstack((colorchunks, [colorarray[i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]]))
        print(f'{i} {j}')

# flatten and process data for training
def transform_input(chunks: ndarray, ceiling: int) -> ndarray:
    #flatten from (1, x, x, ...) to (1, x^2...)    
    chunks = chunks.reshape(chunks.shape[0], np.prod(chunks.shape[1:])) 
    #clip values from 0-ceiling
    #chunks = np.clip(chunks, 0, ceiling)
    #if chunk has all zeros, remove it
    # newchunk = ndarray(shape=(0, chunks.shape[1]))
    # for chunk in chunks:
    #     if np.sum(chunk) > 0:
    #         newchunk = np.vstack((newchunk, chunk))
    # chunks = newchunk
    #chunks = chunks[chunks.sum(axis=1) != 0]

    #add 1 to all chunks
    #chunks = chunks + 1

    #normalize to 0-1
    #chunks = chunks/(ceiling + 2)


    #assert that chunk values are between 0 and 1
    assert np.all(chunks >= 0)
    assert np.all(chunks <= 1)
    return chunks

# input data
temp_chunks = transform_input(input_chunks['temp'], 40)
rain_chunks = transform_input(input_chunks['rain'], 3000)
elev_chunks = transform_input(input_chunks['elev'], 7000)
# target data
colorchunks = transform_input(colorchunks, 255)

print(f'{len(temp_chunks)} {len(rain_chunks)} {len(elev_chunks)} {len(colorchunks)}')

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
