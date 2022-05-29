# Uses 5 minutes/pixel (4320x2160) equirectengular dataset
# for the month of June
# Temperature and precipitation data from WorldClim
# Topography data from NOAA ETOPO
# Visual imagery from NASA Blue Marble


import numpy as np
import imagecodecs
import rasterio

temp_map = rasterio.open(r'temp_map.tif')
rain_map = rasterio.open(r'rain_map.tif')
elev_map = rasterio.open(r'elev_map.tif')
colorarray = imagecodecs.imread(r'colormap.png')


temp_array = np.array(temp_map.read())[0]
rain_array = np.array(temp_map.read())[0]
elev_array = np.array(elev_map.read())[0]

#replace every value in temp array > 100000 with -1
temp_array = temp_array.clip(min=-1, max=40)
rain_array = rain_array.clip(min=-1, max=3000)
elev_array = elev_array.clip(min=-1, max=7000)

#split temp_map into 32x32 chunks

from numpy import float32
CHUNK_SIZE = 16

temp_chunks = np.zeros((1, CHUNK_SIZE, CHUNK_SIZE), dtype=float32)
rain_chunks = np.zeros((1, CHUNK_SIZE, CHUNK_SIZE), dtype=float32)
elev_chunks = np.zeros((1, CHUNK_SIZE, CHUNK_SIZE), dtype=float32)




input_arrays = {'temp': temp_array, 'rain': rain_array, 'elev': elev_array}
input_chunks = {'temp': temp_chunks, 'rain': rain_chunks, 'elev': elev_chunks}

colorchunks = np.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 3), dtype=float32)
# for i in range(0, len(temp_array), 16):
#     for j in range(0, len(temp_array[i]), 16):
for i in range(0, 128, 16):
    for j in range(0, 128, 16):
        
        #iterate through every key
        # for key in input_arrays:
        #     input_chunks[key] = np.vstack((input_chunks[key], [input_arrays[key][i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]]))

        temp_chunks = np.vstack((temp_chunks, [temp_array[i: i+CHUNK_SIZE, j: j+CHUNK_SIZE]]))
        rain_chunks = np.vstack((rain_chunks, [rain_array[i: i+CHUNK_SIZE, j: j+CHUNK_SIZE]]))
        elev_chunks = np.vstack((elev_chunks, [elev_array[i: i+CHUNK_SIZE, j: j+CHUNK_SIZE]]))
        colorchunks = np.vstack((colorchunks, [colorarray[i: i+CHUNK_SIZE, j: j+CHUNK_SIZE]]))

        print(f'{i} {j}')


#set up convolutional neural network model
import keras
from keras import layers
from keras.layers import Input

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


print(temp_chunks.shape)
print(rain_chunks.shape)
print(elev_chunks.shape)
print(colorchunks.shape)

# Dummy input data

#for each variable, flatten 16x16 chunks into 1x256
temp_chunks = temp_chunks.reshape(temp_chunks.shape[0], temp_chunks.shape[1]*temp_chunks.shape[2])
rain_chunks = rain_chunks.reshape(rain_chunks.shape[0], rain_chunks.shape[1]*rain_chunks.shape[2])
elev_chunks = elev_chunks.reshape(elev_chunks.shape[0], elev_chunks.shape[1]*elev_chunks.shape[2])
colorchunks = colorchunks.reshape(colorchunks.shape[0], colorchunks.shape[1]*colorchunks.shape[2]*colorchunks.shape[3])

#for each variable, normalize each chunk to 0-1
temp_chunks = temp_chunks / 40
rain_chunks = rain_chunks / 3000
elev_chunks = elev_chunks / 7000

# temp_chunks = np.asarray(temp_chunks).astype("float32")
# rain_chunks = np.asarray(rain_chunks).astype("float32")
# elev_chunks = np.asarray(elev_chunks).astype("float32")

# Dummy target data
#colorchunks = np.asarray(colorchunks).astype(np.float32)

print(len(temp_chunks))
print(len(rain_chunks))
print(len(elev_chunks))
print(len(colorchunks))
model.summary()

model.fit(
    {"temp": temp_chunks, "rain": rain_chunks, "elev": elev_chunks},
    {"color": colorchunks},
    epochs=2,
    batch_size=32,
)
