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