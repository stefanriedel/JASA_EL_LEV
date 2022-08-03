import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin

COLOR_PLOTS = True
root_dir = dirname(__file__)
utility_path = pjoin(root_dir, 'Utility')
data_dir = pjoin(root_dir, 'ExperimentData')
# Path to store figures
save_path = pjoin(root_dir, 'Figures')

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

# Load loadspeaker gain map according to 15-deg resolution of 24-loudspeaker setup
with open(pjoin(data_dir, 'approx_map.npy'), 'rb') as f:
    approx_map = np.load(f)
num_stimuli = approx_map.shape[0]
# Load 1-deg exact map
with open(pjoin(data_dir, 'exact_map.npy'), 'rb') as f:
    exact_map = np.load(f)


IC_exact = np.zeros((num_stimuli, num_bands))
IC_approx = np.zeros((num_stimuli, num_bands))

ILD_exact = np.zeros((num_stimuli, num_bands))
ILD_approx = np.zeros((num_stimuli, num_bands))

tau_range = np.arange(int(-fs*0.001), int(fs*0.001))

idx_ref = np.arange(hrir.shape[0])  
C_id = np.identity(idx_ref.shape[0], dtype=float)
C_ones = np.ones((idx_ref.shape[0],idx_ref.shape[0]), dtype=float)
# Define signal covariance matrix
C = C_id

# --- Stimuli ---
# Compute IC and ILD: per stimulus and band, for approx angles (15 deg.) and exact angles (1 deg.)

# Select the subset number of LS (2,4,8, or 12)
subset_name = '12'
if subset_name == '2':
    subset_to_plot = 0 + np.arange(4)
if subset_name == '4':
    subset_to_plot = 4 + np.arange(4)
if subset_name == '8':
    subset_to_plot = 8 + np.arange(4)
if subset_name == '12':
    subset_to_plot = 12 + np.arange(4)


for stim in subset_to_plot:
    #exact 
    w_exact = exact_map[stim,:]
    w_exact = np.tile(w_exact, (int(Nfft/2+1),1)).T
    cross_spec_stim_exact = np.diag(np.dot( np.conj(w_exact*h_L).T , np.dot(C, w_exact*h_R)))
    auto_spec_stim_l_exact = np.diag(np.dot( np.conj(w_exact*h_L).T , np.dot(C, w_exact*h_L)))
    auto_spec_stim_r_exact = np.diag(np.dot( np.conj(w_exact*h_R).T , np.dot(C, w_exact*h_R)))

    #approx
    w_approx = approx_map[stim,:]
    w_approx = np.tile(w_approx, (int(Nfft/2+1),1)).T
    cross_spec_stim_approx = np.diag(np.dot( np.conj(w_approx*h_L).T , np.dot(C, w_approx*h_R)))
    auto_spec_stim_l_approx = np.diag(np.dot( np.conj(w_approx*h_L).T , np.dot(C, w_approx*h_L)))
    auto_spec_stim_r_approx = np.diag(np.dot( np.conj(w_approx*h_R).T , np.dot(C, w_approx*h_R)))
    
    for b in range(num_bands):
        window = gammatone_mag_win[b,:]**2

        # exact
        cross_spec_w = cross_spec_stim_exact*window
        auto_spec_l_w = auto_spec_stim_l_exact*window
        auto_spec_r_w = auto_spec_stim_r_exact*window
        cross_correlation = np.real(np.fft.irfft(cross_spec_w))
        P_l = np.fft.irfft(auto_spec_l_w)[0]
        P_r = np.fft.irfft(auto_spec_r_w)[0]
        IC_exact[stim,b] = np.max(np.abs(cross_correlation[tau_range]))  / np.sqrt(P_l * P_r)        
        ILD_exact[stim,b] = 10*np.log10(P_l / P_r)

        #approx
        cross_spec_w = cross_spec_stim_approx*window
        auto_spec_l_w = auto_spec_stim_l_approx*window
        auto_spec_r_w = auto_spec_stim_r_approx*window
        cross_correlation = np.real(np.fft.irfft(cross_spec_w))
        P_l = np.fft.irfft(auto_spec_l_w)[0]
        P_r = np.fft.irfft(auto_spec_r_w)[0]
        IC_approx[stim,b] = np.max(np.abs(cross_correlation[tau_range]))  / np.sqrt(P_l * P_r)        
        ILD_approx[stim,b] = 10*np.log10(P_l / P_r)

plt.figure(figsize=(5, 3))
labels = ['on-center (exact)', 'off-c: const. sources (exact)', 'off-c: line sources (exact)', 'off-c: point sources (exact)', 
'on-center (approx.)', 'off-c: const. sources (approx.)', 'off-c: line sources (approx.)', 'off-c: point sources (approx.)']

ls = ['-', '-.', '--', (0,(1,1)),'-'] * 2
if COLOR_PLOTS:
    ls = ['-'] * 4 + ['--'] * 4
font_sz = 12
cl = [0.3, 0.4, 0.5, 0.6, 0] * 2
colors = ['cornflowerblue', 'goldenrod', 'olivedrab', 'tab:purple'] * 2

idx = 0
for stim in subset_to_plot:
    clr = [cl[idx],cl[idx],cl[idx]]
    if COLOR_PLOTS:
        clr = colors[idx]
    plt.semilogx(f_c, IC_exact[stim,:], label=labels[idx],linewidth=2.5, color=clr ,  linestyle=ls[idx], alpha=0.5)
    plt.semilogx(f_c, IC_approx[stim,:], label=labels[idx+4],linewidth=2.5, color=clr ,  linestyle=ls[idx+4])
    idx += 1
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

plt.title(subset_name + '-loudspeaker conditions (frontal)')
# Put a legend below current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=False, ncol=1)
plt.savefig(fname=pjoin(save_path, 'IC_exact_vs_experiment_approx_' + subset_name + 'LS.pdf'), bbox_inches='tight')
    

plt.figure(figsize=(5, 3))
idx = 0
for stim in subset_to_plot:
    clr = [cl[idx],cl[idx],cl[idx]]
    if COLOR_PLOTS:
        clr = colors[idx]
    plt.semilogx(f_c, ILD_exact[stim,:], label=labels[idx],linewidth=2.5, color=clr ,  linestyle=ls[idx], alpha=0.5)
    plt.semilogx(f_c, ILD_approx[stim,:], label=labels[idx+4],linewidth=2.5, color=clr ,  linestyle=ls[idx+4])
    idx += 1
plt.ylim(-6,6)
plt.yticks([-6,-3,-1,0,1,3,6])
plt.xlim(50,18000)
plt.xlabel('Frequency in Hz', fontsize=font_sz)
plt.ylabel('ILD in dB', fontsize=font_sz)
plt.grid()
ax = plt.gca()
ax.tick_params(which='minor', length=4)
ax.tick_params(which='major', length=6)

plt.title(subset_name + '-loudspeaker conditions (frontal)')
# Put a legend below current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=False, ncol=1)

plt.savefig(fname=pjoin(save_path, 'ILD_exact_vs_experiment_approx_' + subset_name + 'LS.pdf'), bbox_inches='tight')
