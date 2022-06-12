import keras
import numpy as np
from preprocess import *
from matplotlib import pyplot as plt

model = keras.models.load_model('model.keras')

temp_chunks, rain_chunks, elev_chunks = [get_first_tif(file_path) for file_path in [r'prediction_dataset/temperature', r'prediction_dataset/rainfall', r'prediction_dataset/elevation']]
temp_chunks, rain_chunks, elev_chunks = read_multi_files([temp_chunks, rain_chunks, elev_chunks], [40, 3000, 7000])
temp_chunks, rain_chunks, elev_chunks = [chunks for chunks in split_into_chunks(CHUNK_SIZE, [temp_chunks, rain_chunks, elev_chunks])]
temp_chunks, rain_chunks, elev_chunks = [process_nan_values(chunks) for chunks in [temp_chunks, rain_chunks, elev_chunks]]
temp_chunks, rain_chunks, elev_chunks = [np.expand_dims(chunk, axis=1) for chunk in [temp_chunks, rain_chunks, elev_chunks]]

#predict color output for each chunk
for i in range(len(temp_chunks)):
    pred = model.predict({'temp': temp_chunks[i], 'rain': rain_chunks[i], 'elev': elev_chunks[i]})
    pred = np.resize(pred, (CHUNK_SIZE, CHUNK_SIZE, 3))
    plt.imsave(rf'prediction_dataset/color_output/prediction_{i}.png', pred)