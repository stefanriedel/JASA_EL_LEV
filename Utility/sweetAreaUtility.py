import numpy as np

def getCircularArray(nLS, offset_degree=0):
    phi_ls = np.linspace(0, 360-(360/nLS), nLS) + offset_degree
    x_ls = np.cos(phi_ls / 180.0 * np.pi)
    y_ls = np.sin(phi_ls / 180.0 * np.pi)
    
    return x_ls, y_ls, phi_ls

def getRectangularArray(nLS_lateral, nLS_frontback, array_width=1, array_length=1, off_x=1, off_y=1):
    x_ls = np.concatenate((
    np.ones(nLS_lateral)*-1, 
    np.ones(nLS_lateral), 
    np.linspace(-1+off_x/(nLS_frontback+1),1-off_x/(nLS_frontback+1),nLS_frontback), 
    np.linspace(-1+off_x/(nLS_frontback+1),1-off_x/(nLS_frontback+1),nLS_frontback)
    )) * array_width

    y_ls = np.concatenate((
    np.linspace(-1+off_y/(nLS_lateral+1),1-off_y/(nLS_lateral+1),nLS_lateral), 
    np.linspace(-1+off_y/(nLS_lateral+1),1-off_y/(nLS_lateral+1),nLS_lateral), 
    np.ones(nLS_frontback)*-1,
    np.ones(nLS_frontback)
    )) * array_length

    phi_ls = np.arctan2(y_ls , x_ls) / np.pi * 180.0

    draw_rot = np.concatenate(( 
    np.ones(nLS_lateral)*np.pi, 
    np.ones(nLS_lateral)*0, 
    np.ones(nLS_frontback)*np.pi/2*-1,  
    np.ones(nLS_frontback)*np.pi/2)) 

    return x_ls, y_ls, phi_ls, draw_rot

def computeInterauralCues(h_L, h_R, Cov, Lmda, dirs, freq_window, tau_r):
    LCL = Lmda @ Cov @ Lmda
    auto_spectrum_L = np.sum(np.conj(h_L[dirs,:]) * (LCL @ h_L[dirs,:]), axis=0)
    auto_spectrum_R = np.sum(np.conj(h_R[dirs,:]) * (LCL @ h_R[dirs,:]), axis=0)
    cross_spectrum = np.sum(np.conj(h_L[dirs,:]) * (LCL @ h_R[dirs,:]), axis=0)

    auto_spectrum_L = np.tile(auto_spectrum_L[np.newaxis,:], (freq_window.shape[0],1)) * freq_window
    auto_spectrum_R = np.tile(auto_spectrum_R[np.newaxis,:], (freq_window.shape[0],1)) * freq_window
    cross_spectrum = np.tile(cross_spectrum[np.newaxis,:], (freq_window.shape[0],1)) * freq_window

    #P_L = np.max(np.abs( np.fft.irfft(auto_spectrum_L, axis=1))[:,tau_r], axis=1)
    #P_R = np.max(np.abs( np.fft.irfft(auto_spectrum_R, axis=1))[:,tau_r], axis=1)
    # is equivalent to:
    P_L = np.fft.irfft(auto_spectrum_L, axis=1)[:,0]
    P_R = np.fft.irfft(auto_spectrum_R, axis=1)[:,0]  

    ILD = 10 * np.log10(P_L / P_R)

    cross_correlation = np.fft.irfft(cross_spectrum, axis=1)
    IC = np.max(np.abs(cross_correlation)[:,tau_r], axis=1)  / np.sqrt(P_L * P_R)

    # return mean of cues along frequency bands
    return np.mean(IC), np.mean(ILD)