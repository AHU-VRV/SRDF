import os
import pickle
import numpy

outfile_path = 'fitted_multi/lcr_joints_result/joints/lcr_joints.txt'
with open(outfile_path,'w') as fo:

    for i in range(8000):
        joints_file = 'fitted_multi/lcr_joints_result/image_%d.pkl'%(i+1)
        if os.path.exists(joints_file):
            with open(joints_file,'r') as f:
                joints = pickle.load(f)
                fo.write('%f %f %f\n'%(joints['joint3D'][0],joints['joint3D'][1],joints['joint3D'][2]))