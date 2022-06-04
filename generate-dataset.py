# Uses 5 minutes/pixel (4320x2160) equirectengular dataset
# for the month of June
# Temperature and precipitation data from WorldClim
# Topography data from NOAA ETOPO
# Visual imagery from NASA Blue Marble

import numpy as np
import imagecodecs
import rasterio
from numpy import NaN, float32, ndarray

import keras
from keras import layers
from keras.layers import Input

CHUNK_SIZE = 16
MAP_NAMES = ['temp', 'rain', 'elev']
MAX_VALUES = {'temp':40, 'rain':3000, 'elev':7000}

#Reads from f'{filename}_map.tif', clamps to min and max, and removes NaNs'
#seems to be slower this way, better compare between processing input before and after splitting
# filename - string, name of file to read, relative directory
def readfile(filename: str, min: int, max: int):
    
    if filename.find('.tif') != -1:
        tif = rasterio.open(filename)
        arr = np.array(tif.read(), dtype=np.float32)[0]
        tags = tif.tags(1)

        #get min/max from metadata
        if 'STATISTICS_MINIMUN' in tags: min = float(tags['STATISTICS_MINIMUN'])
        if 'STATISTICS_MAXIMUM' in tags: max = float(tags['STATISTICS_MAXIMUM'])

        #for each element, if in tif.nodatavals, set to NaN
        for value in tif.nodatavals:
            arr[arr==value] = np.nan

    elif filename.find('.png') != -1:
        arr = imagecodecs.imread(filename)

    #arr = np.nan_to_num(arr)
    arr = np.clip(arr, min, max)
    arr = (arr - min)/max
    return arr

#split given 2D matrix into chunks of size chunk_size
#and remove chunks with all zeros (based on first given matrix)
# chunk_size - integer, length and width of each segment of the 2D matrix
# array_list - list of 2D matrices to split
# returns a list of 3D ndarrays of shape (n, chunk_size, chunk_size)
def split_into_chunks(chunk_size: int, input_arrays: list) -> list:

    #prepare list of resulting chunks - each ndarray has n of 2D chunks, and extra dimensions for potential subpixels
    # see: broadcasting
    input_chunks = [np.zeros((1, chunk_size, chunk_size) + arr.shape[2:], dtype=float32) for arr in input_arrays]

    #reduced sample size for test purposes, should be len(input_arrays['temp']) and len(input_arrays['temp'][i]
    #row_start = int(len(input_arrays[0]) / 2)
    #col_start = int(len(input_arrays[0][1]) / 2 + 512)
    
    #should be pointed off the Southeastern United States
    #if using a 4320x2160 equirectengular map, centered at long/lat (0, 0)

    #y-coord/latitude
    row_start = 690 
    #x-coord/longitude
    col_start = 1130

    #cauess issues with row = input_arrays[0]) / 2 + 192
    for row in range(row_start, row_start + 128, chunk_size):
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
    
    #replace NaN with mean values
    #for count, chunk in enumerate(chunks):
    for i in range(len(chunks)):
        chunk = chunks[i]   
        mean = np.nanmean(chunk)

        chunks[i] = np.where(np.isnan(chunk), mean, chunk)

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
    output_color = layers.Dense(CHUNK_SIZE**2 * 3, activation='sigmoid', name="color")(x)

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
