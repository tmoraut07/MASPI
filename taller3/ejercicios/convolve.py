import numpy as np
from scipy import signal


def cnnConvolve(filterDim, numFilters, images, Wc, bc):
#%  cnnConvolve Devuelve el resultado de hacer la convolucion de W y b con
#%  las imagenes de entrada
#%
#% Parametetros:
#%  filterDim - dimension del filtro
#%  numFilters - cantidad de filtros
#%  images - imagenes 2D para convolucionar. Estas imagenes tienen un solo
#%  canal (gray scaled). El array images es del tipo images(r, c, image number)
#%  Wc, bc - Wc, bc para calcular los features
#%         Wc tiene tamanio (filterDim,filterDim,numFilters)
#%         bc tiene tamanio (numFilters,1)
#%
#% Devuelve:
#%  convolvedFeatures - matriz de descriptores convolucionados de la forma
#%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    imageDim = images.shape[0]
    numImages = images.shape[2]
    
    convDim = imageDim - filterDim + 1;
    
    convolvedFeatures = np.zeros((convDim, convDim, numFilters, numImages))
    
    #% Instrucciones:
    #%   Convolucionar cada filtro con cada imagen para obtener un array  de
    #%   tamaño (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
    #%   de modo que convolvedFeatures(imageRow, imageCol, featureNum, imageNum) 
    #%   es el valor del descriptor featureNum para la imagen imageNum en la
    #%   region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
    #%
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            #% convolucion simple de una imagen con un filtro
            convolvedImage = np.zeros((convDim, convDim))
            #% Obtener el filtro (filterDim x filterDim) 
            f = Wc[:,:,filterNum]
    
            #% Obtener la imagen
            im = np.squeeze(images[:, :, imageNum])
    
            #%%% IMPLEMENTACION AQUI %%%
            #% Convolucionar "filter" con "im", y adicionarlos a 
            #% convolvedImage para estar seguro de realizar una convolucion
            #% 'valida'
            #% Girar la matriz dada la definicion de convolucion si es necesario (con conv2 no lo es)
            convolvedImage = signal.convolve(im, f, mode="valid")
    
            #%%% IMPLEMENTACION AQUI %%%
            #% Agregar el bias 
            convolvedImage += bc[numFilters-1]
    
            #%%% IMPLEMENTACION AQUI %%%
            #% Luego, aplicar la funcion sigmoide para obtener la activacion de 
            #% la neurona.
            convolvedFeatures[:, :, filterNum, imageNum] = 1 / (1 + np.exp(-1 * convolvedImage))

    return convolvedFeatures
