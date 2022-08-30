import numpy as np

def symmetrizeCircularHRTF(hrtf, earToKeep='Left'):
    if earToKeep=='Left':
        for i in range(int(hrtf.shape[0]/2+1)):
            hrtf[i,1,:] = hrtf[-i,0,:]
            hrtf[-i,1,:] = hrtf[i,0,:]
    if earToKeep=='Right':
         for i in range(int(hrtf.shape[0]/2+1)):
            hrtf[i,0,:] = hrtf[-i,1,:]
            hrtf[-i,0,:] = hrtf[i,1,:]
    return hrtf