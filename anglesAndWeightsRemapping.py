import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin

# Script to compute signal remapping and weighting for experiment conditions (map.npy)
# Additionally, ideal and (actual) remapped angles are exported for error estimation

root_dir = dirname(__file__)
utility_path = pjoin(root_dir, 'Utility')
data_dir = pjoin(root_dir, 'ExperimentData')
# Path to store figures
save_path = pjoin(root_dir, 'Figures')

# Core Parameters
LS_subsets = [2,4,8,12]                                 # List of Array Subsets for Stimuli Creation
source_types = ['center', 'const', 'line', 'point']     # List of Source Types / Pressure Decay Functions
rotations = ['frontal', 'rotated']

amount_off = 0.5                                        # List of relative off-center positions
rot_angle = -90
angular_distortion = True                       # Flag for incorporating angular distortions
level_distortion = True                         # Flag for incorporating level decays

# Total number of physical, equiangular loudspeakers in experiment
nLS = 24
phiLS = np.linspace(90, -270+(360/nLS), nLS)  
x = np.cos(phiLS / 180.0 * np.pi)
y = np.sin(phiLS / 180.0 * np.pi)
coordLS = np.array([x,y]).transpose()

# Total number of reference directions (HRTF set KU100)
nRef = 360
phiRef = np.linspace(90, -270+(360/nRef), nRef)  
x = np.cos(phiRef / 180.0 * np.pi)
y = np.sin(phiRef / 180.0 * np.pi)
coordRef = np.array([x,y]).transpose()

cond_idx = 0
# Map used for 24 LS experiment
exp_approx_map = np.zeros((32, nLS), dtype=float)
# Maps used for error estimation with 360-deg HRTF set
hrtf_approx_map = np.zeros((32, 360), dtype=float)
hrtf_reference_map = np.zeros((32, 360), dtype=float)

DEBUG_PLOT = False

for rot in rotations:
    for n in LS_subsets:
        for source_type in source_types:
            if source_type == 'center':
                listener = np.array([0, 0], dtype=float)
            else:
                listener = np.array([1, 0], dtype=float)  * amount_off

            if(n != 2 and n != 4):
                phi = np.linspace(90, -270+(360/n), n)  
                x = np.cos(phi / 180.0 * np.pi)
                y = np.sin(phi / 180.0 * np.pi)
                coord = np.array([x,y]).transpose()
            if(n is 2):
                # 2 active should be ~stereo geometry
                phi = np.array([45, -225])
                x = np.cos(phi / 180.0 * np.pi)
                y = np.sin(phi / 180.0 * np.pi)
                coord = np.array([x,y]).transpose()
            if(n is 4):
                # 4 active should be ~quadrophonic geometry
                phi = np.array([45, -45, -135, -225])
                x = np.cos(phi / 180.0 * np.pi)
                y = np.sin(phi / 180.0 * np.pi)
                coord = np.array([x,y]).transpose()

            if rot == 'rotated':
                ra = rot_angle * np.pi / 180
                rot_matrix = np.array([ [np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)] ])
                listener = np.dot(rot_matrix, listener)
                coord = np.dot(rot_matrix, coord.T)
                coord = coord.T

            target = coord - np.tile(listener, (n,1)) # Continue coding here
            target_norm = np.zeros(target.shape)
            for i in range(target.shape[0]):
                if(np.linalg.norm(target[i,:]) > 1e-32):
                    target_norm[i,:] = target[i,:] / np.linalg.norm(target[i,:])

            if DEBUG_PLOT:
                plt.plot(coordLS[:,0], coordLS[:,1], 'x', color='k')
                plt.plot(coord[:,0], coord[:,1], '+', color='b')
                plt.plot(target_norm[:,0], target_norm[:,1], 'D', color='r')
                plt.grid()
                plt.ylim(-1.1, 1.1)
                plt.xlim(-1.1,1.1)
                plt.yticks([-1.0,-0.5,0,0.5,1.0])
                plt.xticks([-1.0,-0.5,0,0.5,1.0])
                plt.axis('equal')
                plt.show()
            
            angular_exp_map = np.zeros(n, dtype=int)                # mapping indices for experiment with 24 LS
            angular_ref_map = np.zeros(n, dtype=int)            # mapping indices with 1 deg. resolution (reference)
            level_map = np.ones(n)                              # linear level factors
            
            # Level
            dist_norm = target / np.min(np.linalg.norm(target, axis=1))
            norms = np.linalg.norm(dist_norm, axis=1) 

            if source_type == 'point':
                level_map = 1 / norms
            if source_type == 'line':
                level_map = 1 / np.sqrt(norms)
            if source_type == 'const':
                level_map = np.ones(n)

            level_map = level_map / np.max(level_map)
            # Angular
            if angular_distortion:
                for idx in range(n):
                    # nearest available mapping of source signal / weight
                    angular_exp_map[idx] = np.argmin(np.linalg.norm(coordLS - np.tile(target_norm[idx,:], (nLS, 1)), axis=1))#
                    angular_ref_map[idx] = np.argmin(np.linalg.norm(coordRef - np.tile(target_norm[idx,:], (nRef, 1)), axis=1))

            
            exp_approx_map[cond_idx, angular_exp_map] = level_map

            hrtf_approx_map[cond_idx, -angular_exp_map*15] = level_map
            hrtf_reference_map[cond_idx, -angular_ref_map] = level_map
            cond_idx += 1


#np.save(pjoin(data_dir, 'map.npy'), exp_approx_map)
np.save(pjoin(data_dir, 'approx_map.npy'), hrtf_approx_map)
np.save(pjoin(data_dir, 'exact_map.npy'), hrtf_reference_map)
            
print('done')
