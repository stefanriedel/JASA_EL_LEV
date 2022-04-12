import json
from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats


root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ExperimentData')
save_path = pjoin(root_dir, 'Figures')

# Load loadspeaker gain map according to listening experiment
# Used for visualization of experiment conditions
with open(pjoin(data_dir, 'map.npy'), 'rb') as f:
    gain_map = np.load(f)

N = 24
ratings_WN = np.zeros((N,32,4))
ratings_LPWN = np.zeros((N,32,4))
all_ratings_WN = np.zeros((N,32))
all_ratings_LPWN = np.zeros((N,32))

DifferenceGrade = True

for subj in range(N):
    f_location = pjoin(data_dir, 'subject' + str(subj+1) + '.json')

    f = open(f_location)

    data = json.load(f)

    results = data['Results']
    parts = results['Parts']

    part_WN = parts[0]
    part_LPWN = parts[1]

    trials_WN = part_WN['Trials']
    trials_LPWN = part_LPWN['Trials']


    for idx in range(len(trials_WN)):
        ratings_WN[subj,idx,:] = trials_WN[idx]['Ratings']
        ratings_LPWN[subj,idx,:] = trials_LPWN[idx]['Ratings']

    if DifferenceGrade:
        # Difference between rating of hidden reference and stimulus
        all_ratings_WN[subj, :] = np.ones(32)*100 - (ratings_WN[subj,:,1] - ratings_WN[subj,:,2])
        all_ratings_LPWN[subj, :] = np.ones(32)*100 - (ratings_LPWN[subj,:,1] - ratings_LPWN[subj,:,2])
    else:
        # Absolute stimulus ratings
        all_ratings_WN[subj, :] = ratings_WN[subj,:,2]
        all_ratings_LPWN[subj, :] = ratings_LPWN[subj,:,2]


medians_WN = np.median(all_ratings_WN, axis=0)
medians_LPWN = np.median(all_ratings_LPWN, axis=0)

# Bootstrap Median CIs
data = (all_ratings_WN + np.tile((np.random.rand(24)-0.5)*0.000001, (32,1)).T, )
bstrap = stats.bootstrap(data=data, statistic=np.median, method='BCa', confidence_level=0.95)
median_conf_int_WN = bstrap.confidence_interval

data = (all_ratings_LPWN + np.tile((np.random.rand(24)-0.5)*0.000001, (32,1)).T, )
bstrap = stats.bootstrap(data=data, statistic=np.median, method='BCa', confidence_level=0.95)
median_conf_int_LPWN = bstrap.confidence_interval


# Plot statistical results
indizes_fr = np.arange(0,16,4)
indizes_rot = np.arange(16,32,4)

mpl.rcParams['lines.markersize'] = 6
pl_off = -0.05
cyl_off = -0.025
p_off = 0.025
c_off = 0.0

num_layouts = 4
num_geometries = 7
num_stim_types = 2
rotation_offs = 16+1

medians_NLS = np.zeros((num_layouts, num_geometries, num_stim_types))
median_conf_int_NLS_low = np.zeros((num_layouts, num_geometries, num_stim_types))
median_conf_int_NLS_high = np.zeros((num_layouts, num_geometries, num_stim_types))


for idx in range(num_layouts):
    indices = np.hstack((np.array([idx*4 + np.arange(4)]), np.array([idx*4+rotation_offs + np.arange(3)])))
    medians_NLS[idx,:,0] = medians_WN[indices]
    medians_NLS[idx,:,1] = medians_LPWN[indices]

    median_conf_int_NLS_low[idx,:,0] = median_conf_int_WN.low[indices]
    median_conf_int_NLS_high[idx,:,0] = median_conf_int_WN.high[indices]

    median_conf_int_NLS_low[idx,:,1] = median_conf_int_LPWN.low[indices]
    median_conf_int_NLS_high[idx,:,1] = median_conf_int_LPWN.high[indices]


ratio = 1.0
f, axes = plt.subplots(int(2*num_layouts),num_geometries, gridspec_kw={'height_ratios': [ratio,1,ratio,1,ratio,1,ratio,1], 'hspace' : 0.1, 'wspace' : 0.05})
f.set_size_inches(8.25, 10.7)
xaxis = np.array([0.33,0.66])


nLS = 24
phi = np.linspace(90, -270+(360/nLS), nLS)  
r = 0.75+0.125

t = np.arange(1/8, 1, 1/4) * 2*np.pi
x_square = np.cos(t) * 0.1
y_square = np.sin(t) * 0.1
x_tri = np.array([0,1,-1]) * 0.15
y_tri = np.array([0,1,1]) * 0.15
square = np.array([x_square, y_square]).transpose()
tri = np.array([x_tri, y_tri]).transpose()

sz = 1.25
square = square * sz
tri = tri * sz

phi = phi / 180 * np.pi
x = np.cos(phi)*r
y = np.sin(phi)*r
ls = np.array([x,y]).transpose()
alpha = -np.pi/2 - phi

rot_mat = np.zeros((phi.shape[0], 2, 2))
for n in range(phi.shape[0]):
    rot_mat[n,:,:] = np.array([[np.cos(alpha[n]),-np.sin(alpha[n])],[np.sin(alpha[n]), np.cos(alpha[n])]])

squares = np.zeros((phi.shape[0],4,2))

for n in range(phi.shape[0]):
    squares[n,:,:] = np.dot(square, rot_mat[n,:,:])


# Loudspeaker level pictograms, fill maps
fill_map = np.zeros((num_layouts, num_geometries, 24))
for idx in range(num_layouts):
    indices = np.hstack((np.array([idx*4 + np.arange(4)]), np.array([idx*4+rotation_offs + np.arange(3)])))
    fill_map[idx,:,:] = gain_map[indices,:] 

mks = ['s',  'o']
mks_clr = ['w', 'k']

# Transform fill_map into dB range with 0dB = black
dB_range = 10
fill_map[fill_map<0.001] = 0.001
fill_map = (20*np.log10(fill_map) + dB_range) / dB_range
fill_map = fill_map.clip(min=0)

titles = ['on-center', 'const. sources', 'line sources', 'point sources', 'const. sources', 'line sources', 'point sources']

font_sz = 8

for idx in range(num_layouts): 
    for n in range(num_geometries):
        plt.setp(axes[2*idx,n], xticks=[0.33,0.66], xticklabels=['',''])

        if n >= 1:
            axes[2*idx,n].sharex(axes[2*idx,0])
            plt.setp(axes[2*idx,n], yticks=[], yticklabels=[])
        else:
            plt.setp(axes[2*idx,n], yticks=[0,25,50,75,100], yticklabels=['-4' , '-3', '-2', '-1', '0'])
            axes[2*idx,n].set_ylabel('Difference grade', fontsize=font_sz )
            plt.setp(axes[2*idx,n].get_yticklabels(), fontsize=font_sz)


        # Layout loudspeaker pictograms
        plt.setp(axes[2*idx+1,n], xticks=[], xticklabels=[])
        plt.setp(axes[2*idx+1,n], yticks=[], yticklabels=[])
        plt.setp(axes[2*idx+1,n].get_yticklabels(), visible=False)
        plt.setp(axes[2*idx+1,n].get_xticklabels(), visible=False)

        

        axes[2*idx,n].set_ylim(0,100)
        axes[2*idx,n].set_xlim(0,1)
        axes[2*idx,n].set_yticks([0,25,50,75,100])
        axes[2*idx,n].grid(axis='y')
            
        
        for sidx in range(2):
            asymmetric_error_CI = [ [medians_NLS[idx,n,sidx]-median_conf_int_NLS_low[idx,n,sidx]], [median_conf_int_NLS_high[idx,n,sidx]-medians_NLS[idx,n,sidx] ]]
            axes[2*idx,n].errorbar(xaxis[sidx], medians_NLS[idx ,n, sidx] , capsize=2.0 ,linestyle='none', xerr=0, yerr=asymmetric_error_CI ,color='k', zorder=2)
            axes[2*idx,n].plot(xaxis[sidx], medians_NLS[idx,n,sidx], mks[sidx], fillstyle='full', markerfacecolor=mks_clr[sidx], markeredgecolor='k', zorder=3, label='data')

        if idx == 0:
            axes[2*idx,n].set_title(titles[n], fontsize=font_sz)



        axes[2*idx+1,n].set_ylim(-1,1)
        axes[2*idx+1,n].set_xlim(-1,1)
        axes[2*idx+1,n].set(adjustable='box', aspect='equal')
        axes[2*idx+1,n].axis('off')

        for j in range(phi.shape[0]):
            clr = 1 - fill_map[idx,n,j]
            axes[2*idx+1,n].fill(squares[j,:,0] + ls[j,0] , squares[j,:,1] + ls[j,1], color=[clr,clr,clr], zorder=4, edgecolor=[0.925,0.925,0.925])


color_bar_ax = f.add_axes([0.25,0.05,0.5,0.01])
norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False)
cmap = mpl.cm.get_cmap('gray', 10)
ticks = [0.0, 0.3, 0.6, 0.9, 1.0]
tick_labels = ['0 dB', '-3 dB', '-6 dB', '-9 dB', '-10 dB']
cbar = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_bar_ax, ticks=ticks, orientation='horizontal')
cbar.ax.set_xticklabels(tick_labels)
cbar.ax.tick_params(labelsize=8)

f.legend(bbox_to_anchor=(0.1, 0.07, 0.8, 1), loc='lower left',mode='expand', numpoints=1, ncol=2, fancybox = True,
        fontsize='small', labels=['white noise', 'lowpass noise'], framealpha=1)

plt.savefig(fname=pjoin(save_path, 'ExpResults_DiffGrade.pdf'), bbox_inches='tight')

print('done')

