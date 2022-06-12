

import imagecodecs
import rasterio
import numpy as np
from numpy import NaN, float32, ndarray
import os 

CHUNK_SIZE = 32

#Reads from f'{filename}_map.tif', clamps to min and max, and removes NaNs'
#seems to be slower this way, better compare between processing input before and after splitting
# filename - string, name of file to read, relative directory
# min/max - floats, min and max values to clamp to, will be ignored if GeoTIFF file with min/max is provided
def readfile(filename: str, **kwargs) -> ndarray:

    min, max = 0, 1
    if 'min' in kwargs: min = kwargs['min']
    if 'max' in kwargs: max = kwargs['max']

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

def read_multi_files(file_paths: list, maxval: list) -> list:
    return_list = [0] * len(file_paths)
    for i in range(len(file_paths)):
        return_list[i] = readfile(file_paths[i], max=maxval[i])
    return return_list


#split given 2D matrix into chunks of size chunk_size
#and remove chunks with all zeros (based on first given matrix)
# chunk_size - integer, length and width of each segment of the 2D matrix
# array_list - list of 2D matrices to split
# returns a list of 3D ndarrays of shape (n, chunk_size, chunk_size)
def split_into_chunks(chunk_size: int, input_arrays: list) -> list:

    #prepare list of resulting chunks - each ndarray has n of 2D chunks, and extra dimensions for potential subpixels
    # see: broadcasting
    input_chunks = [np.zeros((0, chunk_size, chunk_size) + arr.shape[2:], dtype=float32) for arr in input_arrays]

    # (row, col) = (690, 1130) should be pointed off the Southeastern United States
    #if using a 4320x2160 equirectengular map, centered at long/lat (0, 0)

    # row_start = 0
    # row_end = len(input_arrays[0])
    # col_start = 0
    # col_end = len(input_arrays[0][1])

    row_start = 690
    row_end = row_start + 128
    col_start = 1130
    col_end = col_start + 128
    term_output_res = max(int((row_end - row_start)/72), chunk_size)

    #causes issues with row = input_arrays[0]) / 2 + 192
    for row in range(row_start, row_end, chunk_size):

        row_display_buff = ''

        for col in range(col_start, col_end, chunk_size):
            add_char_to_buff = (row - row_start) % term_output_res == 0 and (col - col_start) % term_output_res == 0

            #check if temperature chunk is empty, go to next iteration
            if np.sum(input_arrays[0][row:row + chunk_size, col:col + chunk_size]) == 0:

                if add_char_to_buff:
                    row_display_buff += ' '

                continue
            
            #eles if the chunk is not empty 
            if add_char_to_buff:
                row_display_buff += '@'

            #add chunk to output list
            for i in range(len(input_arrays)):

                    
                input_chunks[i] = np.vstack((
                     input_chunks[i], 
                    [input_arrays[i][row:row + chunk_size, col:col + chunk_size]]
                ))

        if row_display_buff:
            print(row_display_buff + '|')

    return input_chunks


# flatten and process data for training
def flatten_input(chunks: ndarray) -> ndarray:

    #flatten from (1, x, x, ...) to (1, x^2...)    
    chunks = chunks.reshape(chunks.shape[0], np.prod(chunks.shape[1:])) 

    assert np.all(chunks >= 0)
    assert np.all(chunks <= 1)
    return chunks

#replace NaN with mean values
def process_nan_values(chunks: ndarray) -> ndarray:
        
    for i in range(len(chunks)):
        chunk = chunks[i]   
        mean = np.nanmean(chunk)

        chunks[i] = np.where(np.isnan(chunk), mean, chunk)

    return chunks

#return first .tif file in directory
def get_first_tif(dirname: str) -> str:
    for filename in os.listdir(dirname):
        if filename.find('.tif') != -1:
            return rf'{dirname}/{filename}'

def get_first_png(dirname: str) -> str:
    for filename in os.listdir(dirname):
        if filename.find('.png') != -1:
            return rf'{dirname}/{filename}'