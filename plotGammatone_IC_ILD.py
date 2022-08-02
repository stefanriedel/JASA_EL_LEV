import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin

COLOR_PLOTS = True
root_dir = dirname(__file__)
utility_path = pjoin(root_dir, 'Utility')
data_dir = pjoin(root_dir, 'ExperimentData')
# Path to store figures
save_path = pjoin(root_dir, 'Figures')

# Load loadspeaker gain map according to listening experiment
with open(pjoin(data_dir, 'map.npy'), 'rb') as f:
    gain_map = np.load(f)
num_stimuli = gain_map.shape[0]

#Load gammatone magnitude windows, precomputed using the 'pyfilterbank' library
# https://github.com/SiggiGue/pyfilterbank
filename = 'gammatone_erb_mag_windows_nfft_4096_numbands_320.npy'
gammatone_mag_win = np.load(pjoin(utility_path, filename))
Nfft = int((gammatone_mag_win.shape[1]-1) * 2)
num_bands = gammatone_mag_win.shape[0]
filename = 'gammatone_fc_numbands_320_fs_48000.npy'
f_c = np.load(pjoin(utility_path, filename))

# Load the HRIR of KU100 dummy head
hrir = np.load(file='./Utility/HRIR_CIRC360_48kHz.npy')
hrtf = np.fft.rfft(hrir ,n=Nfft,axis=-1)
h_L = hrtf[:,0,:]
h_R = hrtf[:,1,:]
fs = 48000

# Experiment settings
LS = 24
idx_ls = -np.linspace(0,(360/LS) * (LS-1),num=LS, dtype=int)
idx_ref = np.arange(hrir.shape[0])   
C_id = np.identity(idx_ls.shape[0], dtype=float)
C_ones = np.ones((idx_ls.shape[0],idx_ls.shape[0]), dtype=float)
# Define signal covariance matrix
C = C_id

# Init Model containers
IC = np.zeros((num_stimuli,num_bands), dtype=float)
IC_ref = np.zeros(num_bands, dtype=float)

ILD = np.zeros((num_stimuli,num_bands), dtype=float)
ILD_ref = np.zeros(num_bands, dtype=float)

tau_range = np.arange(int(-fs*0.001), int(fs*0.001))

# --- Reference --- 2D diffuse field
cross_spec_ref = np.diag(np.dot( np.conj(h_L[idx_ref,:]).T , h_R[idx_ref,:]))
auto_spec_ref_l =  np.diag(np.dot( np.conj(h_L[idx_ref,:]).T , h_L[idx_ref,:]))
auto_spec_ref_r =  np.diag(np.dot( np.conj(h_R[idx_ref,:]).T , h_R[idx_ref,:]))
for b in range(num_bands):
    window = gammatone_mag_win[b,:]**2
    cross_spec_w = cross_spec_ref*window
    auto_spec_l_w = auto_spec_ref_l*window
    auto_spec_r_w = auto_spec_ref_r*window
        
    cross_correlation = np.real(np.fft.irfft(cross_spec_w))
    P_l = np.max(np.abs( np.fft.irfft(auto_spec_l_w)))
    P_r = np.max(np.abs( np.fft.irfft(auto_spec_r_w)))

    IC_ref[b] = np.max(np.abs(cross_correlation[tau_range]))  / np.sqrt(P_l * P_r)
    ILD_ref[b] = 10*np.log10(P_l / P_r)


# --- Stimuli ---
# Compute IC and ILD per stimulus and band
for stim in range(num_stimuli):
    w = gain_map[stim,:]
    w = np.tile(w, (int(Nfft/2+1),1)).T
    cross_spec_stim = np.diag(np.dot( np.conj(w*h_L[idx_ls,:]).T , np.dot(C, w*h_R[idx_ls,:])))
    auto_spec_stim_l = np.diag(np.dot( np.conj(w*h_L[idx_ls,:]).T , np.dot(C, w*h_L[idx_ls,:])))
    auto_spec_stim_r = np.diag(np.dot( np.conj(w*h_R[idx_ls,:]).T , np.dot(C, w*h_R[idx_ls,:])))
    
    for b in range(num_bands):
        window = gammatone_mag_win[b,:]**2
        cross_spec_w = cross_spec_stim*window
        auto_spec_l_w = auto_spec_stim_l*window
        auto_spec_r_w = auto_spec_stim_r*window
            
        cross_correlation = np.real(np.fft.irfft(cross_spec_w))
        #P_l = np.max(np.abs( np.fft.irfft(auto_spec_l_w)[tau_range]))
        #P_r = np.max(np.abs( np.fft.irfft(auto_spec_r_w)[tau_range]))
        # is equivalent to:
        P_l = np.fft.irfft(auto_spec_l_w)[0]
        P_r = np.fft.irfft(auto_spec_r_w)[0]

        IC[stim,b] = np.max(np.abs(cross_correlation[tau_range]))  / np.sqrt(P_l * P_r)        
        ILD[stim,b] = 10*np.log10(P_l / P_r)


plt.figure(figsize=(5, 3))
subset_to_plot = [12,13,14,15]
labels = ['on-center', 'off-c: const. sources', 'off-c: line sources', 'off-c: point sources']
ls = ['-', '-.', '--', (0,(1,1)),'-']
if COLOR_PLOTS:
    ls = ['-', '--', '-','-.', (0, (5, 1))]
font_sz = 12
cl = [0.3, 0.4, 0.5, 0.6, 0]
colors = ['cornflowerblue', 'goldenrod', 'olivedrab', 'tab:purple']

idx = 0
for stim in subset_to_plot:
    clr = [cl[idx],cl[idx],cl[idx]]
    if COLOR_PLOTS:
        clr = colors[idx]
    plt.semilogx(f_c, IC[stim,:], label=labels[idx],linewidth=2.5, color=clr ,  linestyle=ls[idx])
    idx += 1
clr = [cl[idx],cl[idx],cl[idx]]
plt.semilogx(f_c, IC_ref, label='diffuse field', linewidth=2.5, linestyle=ls[idx], color=clr)
plt.ylim(0,1.2)
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.xlim(50,18000)
plt.legend(loc='upper right', framealpha=1, handlelength=2.5)
plt.xlabel('Frequency in Hz', fontsize=font_sz)
plt.ylabel('IC', fontsize=font_sz)
plt.grid()
ax = plt.gca()
ax.tick_params(which='minor', length=4)
ax.tick_params(which='major', length=6)
plt.savefig(fname=pjoin(save_path, 'IC.pdf'), bbox_inches='tight')
    

plt.figure(figsize=(5, 3))
idx = 0
for stim in subset_to_plot:
    clr = [cl[idx],cl[idx],cl[idx]]
    if COLOR_PLOTS:
        clr = colors[idx]
    plt.semilogx(f_c, ILD[stim,:], label=labels[idx], linewidth=2.5, linestyle=ls[idx], color=clr)
    idx += 1
clr = [cl[idx],cl[idx],cl[idx]]
plt.semilogx(f_c, ILD_ref, label='diffuse field', linewidth=2.5, linestyle=ls[idx], color=clr)
plt.ylim(-6,6)
plt.yticks([-6,-3,-1,0,1,3,6])
plt.xlim(50,18000)
plt.xlabel('Frequency in Hz', fontsize=font_sz)
plt.ylabel('ILD in dB', fontsize=font_sz)
plt.grid()
ax = plt.gca()
ax.tick_params(which='minor', length=4)
ax.tick_params(which='major', length=6)
plt.savefig(fname=pjoin(save_path, 'ILD.pdf'), bbox_inches='tight')


print('done')