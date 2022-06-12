import keras
import numpy as np
from preprocess import *
from numpy import NaN, float32, ndarray
from matplotlib import pyplot as plt

#MAP_NAMES = ['temp', 'rain', 'elev']

model = keras.models.load_model('model.keras')
input_arrays = \
{
    'temp': readfile(get_first_tif(r'prediction_dataset/temperature'), 0 , MAX_VALUES['temp']),    
    'rain': readfile(get_first_tif(r'prediction_dataset/rainfall'), 0 , MAX_VALUES['rain']),
    'elev': readfile(get_first_tif(r'prediction_dataset/elevation'), 0 , MAX_VALUES['elev']),
}

temp_chunks, rain_chunks, elev_chunks = \
    [flatten_input(chunks) for chunks in split_into_chunks(CHUNK_SIZE, list(input_arrays.values()))]

#add a third dimension to each variable, for training purposes
temp_chunks, rain_chunks, elev_chunks = \
    [np.expand_dims(chunk, axis=1) for chunk in [temp_chunks, rain_chunks, elev_chunks]]

temp_chunks, rain_chunks, elev_chunks = \
    [np.expand_dims(chunk, axis=1) for chunk in [temp_chunks, rain_chunks, elev_chunks]]

data_dict = zip(MAP_NAMES, [temp_chunks, rain_chunks, elev_chunks]) 
for i in range(len(temp_chunks)):

    #predicts the color of the chunk
    #after adding another dimension
    #(training data also got a blank dimension added to it)
    # pred = model.predict({
    #         key:value
    #         for key, value
    #         in data_dict
    # })

    pred = model.predict({'temp': temp_chunks[i], 'rain': rain_chunks[i], 'elev': elev_chunks[i]})


    pred = np.resize(pred, (CHUNK_SIZE, CHUNK_SIZE, 3))
    #imagecodecs.imwrite(rf'prediction_dataset/prediction_{i}.png', pred)


    plt.imsave(rf'prediction_dataset/color_output/prediction_{i}.png', pred)

"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#show 3x2 images in a grid
#fig, axs = plt.subplots(3, 2, figsize=(10, 10))
fig = plt.figure(figsize=(10., 10.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                 axes_pad=0.5,  # pad between axes in inch.
                 )


for count, ax in enumerate(grid):
    
    if count >= 5:
        break
    chunk = [temp_chunks, rain_chunks, elev_chunks, colorchunks, [None, pred]][count][1]
    if 256 in chunk.shape:
        chunk = np.reshape(chunk, (16,16))
    elif 768 in chunk.shape:
        chunk = np.reshape(chunk, (16, 16, 3))
    ax.imshow(chunk)
    ax.set_title(f"{((MAP_NAMES + ['Colormap (Actual)', 'Colormap (Predicted)'])[count])}")

plt.show()
"""