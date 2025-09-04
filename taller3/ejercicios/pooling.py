import numpy as np
from scipy import signal

def cnnPool(poolDim, convolvedFeatures):
#%  cnnPool Pools los descriptores provenientes de la convolucion
#%  La funcion usa el Pool PROMEDIO
#% Parametros:
#%  poolDim - dimension de la regiom de pool
#%  convolvedFeatures - los descriptores a realizar el pool 
#%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
#%
#% Devuelve:
#%  pooledFeatures - matriz de los features agrupados
#%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
#%     

    numImages = convolvedFeatures.shape[3]
    numFilters = convolvedFeatures.shape[2]
    convolvedDim = convolvedFeatures.shape[0] # me sirve usar uno ya que las imagenes son cuadradas

    px, py = int(convolvedDim / poolDim), int(convolvedDim / poolDim)
    pooledFeatures = np.zeros((px, py, numFilters, numImages))

#% Instrucciones:
#%   Realizar el pool de los features en regiones de tamaño poolDim x poolDim,
#%   para obtener la matriz pooledFeatures de 
#%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 

#%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) es el valor 
#%   del descriptor featureNum de la imagen imageNum agrupada sobre la 
#%   region (poolRow, poolCol). 
#%   

#%%% IMPLEMENTAR AQUI %%%
    aux_filter = np.ones((poolDim, poolDim, numFilters, numImages)) / poolDim
    pooledFeatures = signal.convolve(pooledFeatures, aux_filter)
    return pooledFeatures

