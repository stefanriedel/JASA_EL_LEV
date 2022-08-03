import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from os.path import dirname, join as pjoin
from Utility.sweetAreaUtility import getCircularArray, getRectangularArray, computeInterauralCues
from joblib import Parallel, delayed

if __name__ == '__main__':
    # Define directory for saving figures
    root_dir = dirname(__file__)
    save_path = pjoin(root_dir, 'Figures')
    utility_path = pjoin(root_dir, 'Utility')

    #Load gammatone magnitude windows, precomputed using the 'pyfilterbank' library
    # https://github.com/SiggiGue/pyfilterbank
    filename = 'gammatone_erb_mag_windows_nfft_1024_numbands_320.npy'
    gammatone_mag_win = np.load(pjoin(utility_path, filename))
    Nfft = int((gammatone_mag_win.shape[1]-1) * 2)
    num_bands = gammatone_mag_win.shape[0]
    filename = 'gammatone_fc_numbands_320_fs_48000.npy'
    f_c = np.load(pjoin(utility_path, filename))

    # ERB scale requires using every 8th of the 320 windows
    f_c = f_c[::8]
    gammatone_mag_win = gammatone_mag_win[::8] 
    USE_GAMMATONE_WINDOWS = True

    # Load HRTF set
    hrir = np.load(file='./Utility/HRIR_CIRC360_48kHz.npy')
    fs = 48000
    #Nfft = 1024
    f = np.linspace(0,fs/2, num=int(Nfft/2+1))
    hrtf = np.fft.rfft(hrir, n=Nfft, axis=-1) 
    h_L = hrtf[:,0,:]
    h_R = hrtf[:,1,:]

    # Define meshgrid resolution and simulation area
    res = 40    
    array_radius = 1
    area_len = 1.1

    # Define head rotations used for evaluation of IC and ILD
    rotations = np.array([0,30,60,90,120,150,-180,-150,-120,-90,-60,-30])   # set between -180 and 180
    # -> if needed, reduce res and rotations to speed up the computation time
    # alternatively, set USE_GAMMATONE_WINDOWS = False, to use single broadband window

    # Specify simulated layout
    RECTANGULAR_ARRAY = False
    if not RECTANGULAR_ARRAY:
        # get coordinates for circular array with nLS sources
        nLS = 8
        x_ls, y_ls, phi_ls = getCircularArray(nLS=nLS, offset_degree=45)
    else:
        # get coordinates for rectangular array
        array_width = 0.6
        array_length = 1.0
        x_ls, y_ls, phi_ls, draw_rot = getRectangularArray(nLS_lateral=5, nLS_frontback=3, 
        array_width=array_width, array_length=array_length, off_x=2, off_y=1)

    #Define source signal covariance matrix, default is unit matrix
    Cov = np.eye(phi_ls.size)
    #Cov = np.ones((phi_ls.size,phi_ls.size))

    # Stack source coordinates
    ls = np.array([x_ls,y_ls]).transpose()
    ls = ls * array_radius

    # Define source model list
    source_models = ['const','line','point']
    num_source_models = len(source_models)

    # Create listener meshgrid
    x = np.linspace(-area_len, area_len, res) * array_radius
    y = np.linspace(-area_len, area_len, res) * array_radius
    [listener_X, listener_Y] = np.meshgrid(x,y)
    listener = np.vstack( (listener_X.flatten(), listener_Y.flatten()) ).transpose()

    num_rotations = rotations.size
    num_listener_pos = listener.shape[0]
    num_source_pos = ls.shape[0]

    theta = np.zeros((num_source_pos, 2, num_listener_pos))

    ls = np.tile(ls[:,:,np.newaxis], (1,1,num_listener_pos))
    listener = listener.transpose()
    listener = np.tile(listener[np.newaxis, :, :], (num_source_pos, 1, 1))

    theta = ls - listener

    r = np.linalg.norm(theta, axis=1)
    r = np.tile(r[:,np.newaxis,:], (1,2,1))

    theta_norm = theta  / r
    r_norm = r / np.tile(np.min(r, axis=0)[np.newaxis,:,:] , (num_source_pos,1,1))
    r_norm = r_norm[:,0,:]  #normalized relative distance to sources (remove tiling) 

    source_phi = np.arctan2(theta_norm[:,1,:] , theta_norm[:,0,:]) / np.pi * 180.0 #+ 90.0
    source_phi = source_phi.astype(np.int32)

    LEV_ILD = np.zeros((num_source_models,num_listener_pos, num_rotations))
    LEV_IC = np.zeros((num_source_models,num_listener_pos,num_rotations))

    if USE_GAMMATONE_WINDOWS:
        # computes mean IC/ILD across gammatone windows, more costly
        low_lim = int(np.where(f_c>=400)[0][0])
        up_lim = int(np.where(f_c>=6000)[0][0])
        freq_window = gammatone_mag_win[low_lim:up_lim,:]
    else:
        # define frequency limits for a single binary window
        low_lim = int(np.where(f>=400)[0][0])
        up_lim = int(np.where(f>=6000)[0][0])
        freq_window = np.zeros(int(Nfft/2+1))
        freq_window[low_lim:up_lim] = 1
        freq_window = freq_window[np.newaxis,:]

    # tau search range, typically +- 1 millisecond
    tau_r = np.arange(-int(fs/1000),int(fs/1000))

    def mainLoopSourceModels(s):
        LEV_IC_tmp = np.zeros((num_listener_pos, num_rotations))
        LEV_ILD_tmp = np.zeros((num_listener_pos, num_rotations))

        source_model = source_models[s]

        if source_model == 'point':
            w_r = 1 / r_norm
        if source_model == 'line':
            w_r = 1 / r_norm**(0.5)
        if source_model == 'const':
            w_r = 1 / np.ones(r_norm.shape)

        for rot in range(num_rotations):
            for p in range(num_listener_pos):
                idcs = source_phi[:,p] + rotations[rot]
                Lmda = np.diag(w_r[:,p])
                IC, ILD = computeInterauralCues(h_L, h_R, Cov, Lmda, idcs, freq_window, tau_r)
                LEV_IC_tmp[p,rot] = 1 - IC
                LEV_ILD_tmp[p,rot] = -np.abs(ILD)

        return LEV_IC_tmp, LEV_ILD_tmp

    print('Main simulation loop started... \n')
    result_lists = Parallel(n_jobs=3)(delayed(mainLoopSourceModels)(s) for s in range(num_source_models))
    print('Plot is being saved to .\Figures now.')

    res_arr = np.asarray(result_lists)

    for s in range(num_source_models):
        LEV_IC[s,:,:] = res_arr[s,0,:,:]
        LEV_ILD[s,:,:] = res_arr[s,1,:,:]

    LEV_ILD = np.min(LEV_ILD, axis=-1)
    LEV_IC = np.min(LEV_IC, axis=-1)

    LEV_ILD = np.reshape(LEV_ILD, (num_source_models, res,res))
    LEV_IC = np.reshape(LEV_IC, (num_source_models, res,res))

    # Plotting the computed values
    scale = 0.8
    fig, axes = plt.subplots(nrows=2, ncols=num_source_models, figsize=(12*scale,8*scale))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    ILD_lvls = np.array([-10,-6,-3,-1,0])
    IC_lvls = np.array([0.0, 0.2, 0.4, 0.6, 1.0])

    LEV_ILD = np.clip(LEV_ILD, a_min=-9.9, a_max=0.0)

    font_sz = 18

    cbar_ticklabels = [['10','6','3','1','0'],  ['1.0', '0.8' , '0.6', '0.4', '0.0'] ]
    cbar_pos_size = [[0.92, 0.51, 0.01, 0.37], [0.92, 0.11, 0.01, 0.37]]

    bmap = colors.ListedColormap(["black", "black", "black", "black"])

    titles = [ 'const. sources', 'line sources', 'point sources']

    vmins = [-10, 0.0]
    vmaxs = [0.0, 1.0]

    clrs = ['0.25', '0.5', '0.75', '1.0']

    for r in range(2):
        for c in range(num_source_models):
            if r == 0:
                LEV = LEV_ILD[c,:]
                lvls = ILD_lvls
                axes[r,c].set_title(titles[c], fontsize=font_sz)
                if c == 0:
                    axes[r,c].set_ylabel('ILD in dB', fontsize=font_sz)
            if r == 1:
                LEV = LEV_IC[c,:]
                lvls = IC_lvls
                if c == 0:
                    axes[r,c].set_ylabel('IC', fontsize=font_sz)
            pcm = axes[r,c].contourf(listener_X,listener_Y, LEV, levels=lvls, colors=clrs, zorder=1, vmin=vmins[r], vmax=vmaxs[r])

            if c == num_source_models-1:
                color_bar_ax = fig.add_axes(cbar_pos_size[r])
                cbar = fig.colorbar(pcm, cax=color_bar_ax)
                cbar.ax.set_yticklabels(cbar_ticklabels[r])
                cbar.ax.tick_params(labelsize=14)

            axes[r,c].contour(listener_X,listener_Y, LEV, levels=lvls, cmap=bmap, linewidths=0.5, zorder=2)
            axes[r,c].set_xlim(-area_len, area_len)
            axes[r,c].set_ylim(-area_len, area_len)
            axes[r,c].set_xticks([])
            axes[r,c].set_yticks([])
            
            # Source / Loudspeaker icon drawing
            t = np.arange(1/8, 1, 1/4) * 2*np.pi
            x_square = np.cos(t) * 0.1
            y_square = np.sin(t) * 0.1
            x_tri = np.array([0,1,-1]) * 0.15
            y_tri = np.array([0,1,1]) * 0.15
            square = np.array([x_square, y_square]).transpose()
            tri = np.array([x_tri, y_tri]).transpose()

            sz = 0.5
            square = square * sz
            tri = tri * sz

            phi = phi_ls / 180 * np.pi
            ls = np.array([x_ls,y_ls]).transpose()
            #tr = ls_coord
            alpha = -np.pi/2 - phi

            if RECTANGULAR_ARRAY:
                alpha = -np.pi/2 - draw_rot

            rot_mat = np.zeros((phi.shape[0], 2, 2))
            for n in range(phi.shape[0]):
                rot_mat[n,:,:] = np.array([[np.cos(alpha[n]),-np.sin(alpha[n])],[np.sin(alpha[n]), np.cos(alpha[n])]])

            squares = np.zeros((phi.shape[0],4,2))
            tris = np.zeros((phi.shape[0],3,2))

            for n in range(phi.shape[0]):
                squares[n,:,:] = np.dot(square, rot_mat[n,:,:])
                tris[n,:,:] = np.dot(tri, rot_mat[n,:,:])

            for n in range(phi.shape[0]):
                axes[r,c].fill(tris[n,:,0] + ls[n,0] , tris[n,:,1] + ls[n,1], color='w', zorder=3, edgecolor='k')
                axes[r,c].fill(squares[n,:,0] + ls[n,0] , squares[n,:,1] + ls[n,1], color='w', zorder=4, edgecolor='k')

    if USE_GAMMATONE_WINDOWS:
        add_string = '_ERB'
    else:
        add_string = ''

    if not RECTANGULAR_ARRAY:
        plt.savefig(fname=pjoin(save_path, str(nLS) + 'LS_sweet_area' + add_string + '.pdf'), bbox_inches='tight')
    else:
        plt.savefig(fname=pjoin(save_path, 'RECT_' + str(int(array_width*100)) + 'wide' + 
        str(int(array_length*100)) + 'long_' + str(phi_ls.size) + 'LS_sweet_area' + add_string + '.pdf'), bbox_inches='tight')

