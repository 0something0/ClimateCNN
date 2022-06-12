import keras
import numpy as np
from preprocess import *
from matplotlib import pyplot as plt

model = keras.models.load_model('model.keras')
input_arrays = \
{
    'temp': readfile(get_first_tif(r'prediction_dataset/temperature'), 0 , MAX_VALUES['temp']),    
    'rain': readfile(get_first_tif(r'prediction_dataset/rainfall'), 0 , MAX_VALUES['rain']),
    'elev': readfile(get_first_tif(r'prediction_dataset/elevation'), 0 , MAX_VALUES['elev']),
}

temp_chunks, rain_chunks, elev_chunks = \
    [process_nan_values(chunks) for chunks in split_into_chunks(CHUNK_SIZE, list(input_arrays.values()))]

#add a third dimension to each variable, reshaping each chunk from (CHUNK_SIZE**2,) to match (None, CHUNK_SIZE**2)
temp_chunks, rain_chunks, elev_chunks = \
    [np.expand_dims(chunk, axis=1) for chunk in [temp_chunks, rain_chunks, elev_chunks]]

#predict color output for each chunk
data_dict = zip(MAP_NAMES, [temp_chunks, rain_chunks, elev_chunks]) 
for i in range(len(temp_chunks)):

    pred = model.predict({'temp': temp_chunks[i], 'rain': rain_chunks[i], 'elev': elev_chunks[i]})
    pred = np.resize(pred, (CHUNK_SIZE, CHUNK_SIZE, 3))

    plt.imsave(rf'prediction_dataset/color_output/prediction_{i}.png', pred)