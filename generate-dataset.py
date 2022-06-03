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
MAX_VALUES = {'temp':40, 'rain':3000, 'elev':7000}

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

#get resulting shape of matrix if splitting into target_shape within some axis
# origin_shape - the shape of the original ndarray
# target_shape - the desired shape of ndarray in a limited set of dimensions
#                  should be a shorter tuple than origin_shape
# returns a tuple of the resulting shape
def calculate_shape(origin_shape: tuple, target_shape: tuple) -> tuple:
    assert len(origin_shape) >= len(target_shape)

    return target_shape + origin_shape[len(target_shape):]



#split given 2D matrix into chunks of size chunk_size
#and remove chunks with all zeros (based on first given matrix)
# chunk_size - integer, length and width of each segment of the 2D matrix
# array_list - list of 2D matrices to split
# returns a list of 3D ndarrays of shape (n, chunk_size, chunk_size)
def split_into_chunks(chunk_size: int, input_arrays: list) -> list:

    input_chunks = [np.zeros((1, chunk_size, chunk_size) + arr.shape[2:], dtype=float32) for arr in input_arrays]

    #reduced sample size for test purposes, should be len(input_arrays['temp']) and len(input_arrays['temp'][i]
    row_start = int(len(input_arrays[0]) / 2)
    col_start = int(len(input_arrays[0][1]) / 2 + 512)

    for row in range(row_start, + 128, chunk_size):
        for col in range(col_start, col_start + 128, chunk_size):

            #check if temp chunk is empty, go to next iteration
            if np.sum(input_arrays[0][row:row + chunk_size, col:col + chunk_size]) == 0:
                continue

            #add chunk to output list
            for i in range(len(input_arrays)):
                input_chunks[i] = np.vstack((
                     input_chunks[i], 
                    [input_arrays[i][row:row + chunk_size, col:col + chunk_size]]
                ))

            print(f'{row} {col}')
            
    return input_chunks


# flatten and process data for training
def flatten_input(chunks: ndarray) -> ndarray:
    #flatten from (1, x, x, ...) to (1, x^2...)    
    chunks = chunks.reshape(chunks.shape[0], np.prod(chunks.shape[1:])) 
    
    assert np.all(chunks >= 0)
    assert np.all(chunks <= 1)
    return chunks


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


input_arrays = {key: readfile(rf'{key}_map.tif', 0 , MAX_VALUES[key]) for key in MAP_NAMES}
colorarray = readfile(r'colormap.png', 0 ,255)

temp_chunks, rain_chunks, elev_chunks, colorchunks = \
    [flatten_input(chunks) for chunks in split_into_chunks(CHUNK_SIZE, list(input_arrays.values()) + [colorarray])]


print(f'{len(temp_chunks)} {len(rain_chunks)} {len(elev_chunks)} {len(colorchunks)}')

model = build_model()
model.summary()
model.fit(
    {"temp": temp_chunks, "rain": rain_chunks, "elev": elev_chunks},
    {"color": colorchunks},
    epochs=2,
    batch_size=32,
)
