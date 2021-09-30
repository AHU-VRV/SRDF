
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
from time import *
from glob import glob
import argparse

import cv2
import numpy as np
import chumpy as ch

from opendr.camera import ProjectPoints
from lib.robustifiers import GMOf
from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation
from smpl_webuser.verts import verts_decorated
from lib.sphere_collisions import SphereCollisions
from lib.max_mixture_prior import MaxMixtureCompletePrior
from render_model import render_model
from scipy.spatial import KDTree

_LOGGER = logging.getLogger(__name__)


# Mapping from kinect joints to SMPL joints.
# 1 Spine mid       6
# 2 Neck            15
# 4 Shoulder left   16
# 5 Elbow left      18
# 6 Wrist left      20
# 8 Shoulder right  17
# 9 Elbow right     19
# 10 Wrist right    21
# 12 Hip left       1
# 13 Knee left      4
# 14 Ankle left     7
# 15 Foot left      10
# 16 Hip right      2
# 17 Knee right     5
# 18 Ankle right    8
# 19 Foot right     11
# 20 Spine Shoulder 12
# 3 Head            added

#Init translation and rotation
#Use four joints: Spine mid, Shoulder left, Shoulder right, Spine Shoulder.
def init_rt(model,j3d,init_pose):

    smpl_ids = [16,17,1,2]
    kin_ids = [4,8,12,16]


    ratio = ch.array([1.0])
    init_trans = ch.array([0,0,0])
    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])

    # diff_Jtr = np.array([Jtr[16] - Jtr[1], Jtr[17] - Jtr[2]])
    # mean_height_Jtr = np.mean(np.sqrt(np.sum(diff_Jtr ** 2, axis=1)))
    #
    # diff_kin = np.array([j2d[4] - j2d[12], j2d[8] - j2d[16]])
    # mean_height_kin = np.mean(np.sqrt(np.sum(diff_kin ** 2, axis=1)))

    ratio = np.linalg.norm(j3d[4]-j3d[8])/np.linalg.norm(Jtr[16]-Jtr[17])
    ratio = 0.9
    a = np.linalg.norm(j3d[12]-j3d[16])/np.linalg.norm(Jtr[1]-Jtr[2])
    b = np.linalg.norm(j3d[4]-j3d[8])/np.linalg.norm(Jtr[16]-Jtr[17])


    ch.minimize(
        {'rt':j3d[kin_ids]-(Jtr[smpl_ids]*ratio+init_trans)},
        x0=[opt_pose[:3],init_trans],
        method='dogleg',
        callback=None,
        options={'maxiter':100,
                 'e_3':.0001,
                 'disp':0})

    Jtr0 = Jtr*ratio

    # Init_trans = (j2d[4]+j2d[8])/2-(Jtr[16]*ratio+Jtr[17]*ratio)/2
    return (init_trans,opt_pose[:3].r,ratio)

#Optimize_on_joints(file_path_number,ratio,model, j2d, hj, init_trans, init_rot, prior, n_betas=n_betas, regs=regs, conf=conf,pyr = pyr)
def optimize_on_joints(file_path_number,
                       ratio,
                       model,
                       j3d,
                       hj,
                       pcv,
                       pcvn,
                       init_trans,
                       init_rot,
                       prior,
                       n_betas=10,
                       regs=None,
                       conf=None,
                       pyr = None):
    """Run the fit for one specific image.
    :param file_path_number: result save path
    :param ratio: scale
    :param model: SMPL model
    :param j3d: 25x3 array of body joints
    :param hj: 38x3 array of hand joints
    :param pcv: the vertexs of kinect point cloud
    :param pcvn: the normals of the vertexs of pc
    :param init_trans: initial translation
    :param init_rot: initial_rot
    :param prior: mixture of gaussians pose prior
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param conf: the confidence values
    :param pyr: the head pose data
    """
    t0=time()

    #Define the mapping: kinect joints to smpl joints
    #SMPL does not have a joint for head, instead we use a vertex for the head
    # and append it later.
    kinect_ids = [4,5,6,8,9,10,13,14,17,18,20,2,15,19]
    smpl_ids = [16,18,20,17,19,21,4,7,5,8,12,15]
    left_f = 3365
    right_f = 6765

    #Define the vertex id corresponding to the fingertip
    model_id = [2731, 2314, 2426, 2536, 2653, 6208, 5781, 5899, 5999, 6118]

    # The vertex id for the joint corresponding to the head
    head_id = 411

    # Weights assigned to each joint during optimization;
    base_weights = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,0.95,0.95,0.95], dtype=np.float64)

    # Initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(n_betas)

    # Initialize the pose by using the initial rotation and the
    # pose prior
    init_rot = np.zeros(3)
    init_pose = np.hstack((init_rot, prior.weights.dot(prior.means)))
   # Init_pose = np.hstack((init_rot,np.zeros(69)))
    init_trans = ch.array([0,0.2,0])
    # Instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)
    init_trans = ch.array([0,0,0])
    init_pose = ch.zeros(72)
    sv = verts_decorated(
        trans=init_trans,
        pose=init_pose,
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :n_betas],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs,
        want_Jtr=True)

    # set head pose on the basis of pose[45:48]
    # #
    diff_x = sv[336][1] - sv[384][1]
    diff_y = sv[166][2] - sv[3678][2]
    diff_z = sv[336][0] - sv[384][0]

    point_index = [336, 384, 166, 3678, 336, 3161]
    flatx = ch.array([0,0,1])
    flaty = ch.array([1,0,0])
    flatz = ch.array([0,1,0])
    vector_x = ch.vstack((sv[384][0], sv[336][1], sv[336][2])).reshape(3) - sv[384]
    vector_y = ch.vstack((sv[166][0], sv[3678][1], sv[166][2])).reshape(3) - sv[3678]
    vector_z = ch.vstack((sv[336][0], sv[336][1], sv[3161][2])).reshape(3) - sv[3161]
    theta_init = np.zeros(3)
    #rotation of x
    moldx = ch.sqrt(vector_x[0] ** 2 + vector_x[1] ** 2 + vector_x[2] ** 2)
    cosx = (vector_x[0] * flatx[0] + vector_x[1] * flatx[1] + vector_x[2] * flatx[2]) / moldx
    thetax = ch.arcsin(ch.sqrt(1 - cosx ** 2))
    if(diff_x.r <0):
        theta_init[0] = -thetax.r
    else:
        theta_init[0] = thetax.r
    #rotation of y
    moldy = ch.sqrt(vector_y[0] ** 2 + vector_y[1] ** 2 + vector_y[2] ** 2)
    cosy = (vector_y[0] * flaty[0] + vector_y[1] * flaty[1] + vector_y[2] * flaty[2]) / moldy
    thetay = ch.arcsin(ch.sqrt(1 - cosy ** 2))
    if (diff_y.r < 0):
        theta_init[1] = thetay.r
    else:
        theta_init[1] = -thetay.r
    #rotation of z
    moldz = ch.sqrt(vector_z[0] ** 2 + vector_z[1] ** 2 + vector_z[2] ** 2)
    cosz = (vector_z[0] * flatz[0] + vector_z[1] * flatz[1] + vector_z[2] * flatz[2]) / moldz
    thetaz = ch.arcsin(ch.sqrt(1 - cosz ** 2))
    if (diff_z.r < 0):
        theta_init[2] = -thetaz.r
    else:
        theta_init[2] = thetaz.r

    Jtr = sv.J_transformed[smpl_ids]
    Jtr = ch.vstack((Jtr, sv[left_f]))
    Jtr = ch.vstack((Jtr, sv[right_f]))
    # Update the weights using confidence values
    weights = base_weights# * conf[kinect_ids] if conf is not None else base_weights
    #weights = conf[kinect_ids]
    # data term: distance between observed and estimated joints in 2D
    ##EJ
    obj_j2d = lambda w, sigma: (w * weights.reshape((-1, 1)) * GMOf((j3d[kinect_ids] - Jtr), sigma))

    # mixture of gaussians pose prior
    pprior = lambda w: w * prior(sv.pose)
    # joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10
    my_exp = lambda x: alpha * ch.exp(x)
    ##Ea
    obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])

    if regs is not None:
        # interpenetration term
        ##Esp
        sp = SphereCollisions(
            pose=sv.pose, betas=sv.betas, model=model, regs=regs)
        sp.no_hands = True

    opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                      [1e2, 5 * 1e1, 1e1, .5 * 1e1])


    #w2  0.1  1
    w0 = 1#joint
    w1 = 1#pose
    w2 = 0.5#betas
    objs = {}
    # Skeleton
    for ids in range(10):
        _LOGGER.info('stage_%d\n'%(ids))
        # _LOGGER.info('w0%d  w1%d\n' % (w0,w1))
        #objs = {}

        tmpBetas = ch.zeros(10)
        tmpBetas[:] = betas.r
        if (ids == 0):
            objs['betas'] = w2 * (sv.betas - tmpBetas)
        else:
            objs['betas'] = w2*(sv.betas - tmpBetas) ##Es

        objs['j3d'] = obj_j2d(w0, 100) ##Edata_b
        # w0 += 1

        objs['pose'] = pprior(w1) #Ep
        # w1 -= 0.1
        objs['trans'] = 10 * init_trans
        #
        # if(ids == 8):
        #     w0 = 1
        #     w1 = 0.1

        if regs is not None:
            if ids != 10:
                objs['sph_coll'] = 1e3 * sp  ##Esp

        if ids == 0:
            # w2 -= 1
            free_variables = [sv.betas, sv.pose[:3], sv.trans]
        elif ids < 10:
            free_variables = [sv.betas, sv.pose, sv.trans]
            w0 += 1
            w1 -= 0.1
            w2 -= 0.05

        ch.minimize(
            objs,
            x0=free_variables,
            method='dogleg',
            callback=None,
            options={'maxiter': 300,
                     'e_3': .0001,
                     'disp': 0})

        outmesh_path = './fitted_model_%d/kinect_smpl_test_%d.obj'%(file_path_number,ids)
        with open(outmesh_path, 'w') as fp:
            for v in sv.r:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in sv.f+1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    #
    bias_l = hj[22-22] - sv.J_transformed[22]
    bias_r = hj[37-22] - sv.J_transformed[37]
    hj[0:15] = hj[0:15]-bias_l
    hj[15:29] = hj[15:29] - bias_r
    hj[29:33] = hj[29:33] - bias_l
    hj[33:37] = hj[33:37] - bias_r
    obj_ha = hj - ch.vstack((sv.J_transformed[22:51],sv[model_id])) ##Edata_h
    ch.minimize(
        obj_ha,
        x0=[sv.pose[20:]],
        method='dogleg',
        callback=None,
        options={'maxiter': 100,
                 'e_3': .0001,
                 'disp': 0})

    # Head
    pyr = pyr.astype(float)/180*np.pi
    pyr = ch.array(pyr)
    #pyr = pyr-sv.pose.r[:3]*np.array([-1,1,-1])

    obj_he = {} ##Eh
    obj_he['x'] = (pyr[0] - (thetax * diff_x / ch.absolute(diff_x) - theta_init[0]))
    obj_he['y'] = (pyr[1] - (thetay * -diff_y / ch.absolute(diff_y) - theta_init[1]))
    obj_he['z'] = (pyr[2] - (thetaz * diff_z / ch.absolute(diff_z) - theta_init[2]))
    obj_he['pose'] = pprior(0.1)
    ch.minimize(
        obj_he,
        # x0=[sv.pose[45:48],sv.pose[36:39],sv.pose[18:21],sv.pose[27:30]],
        x0=[sv.pose[45:48], sv.pose[36:39]],
        method='dogleg',
        callback=None,
        options={'maxiter': 100,
                 'e_3': .0001,
                 'disp': 0})

    #Point cloud
    tree = KDTree(pcv.r)
    v_indexs = tree.query(sv.r)[1]
    vcm = pcv[v_indexs] - sv
    obj_pc = []
    obj_pc['data'] = (1 if vcm[3]/(-abs(vcm[3]))==1 else 0) * 0.2 * GMOf((ch.dot(pcvn[v_indexs],sv.T)),0.2)+\
          (1 - 1 if vcm[3]/(-abs(vcm[3]))==1 else 0) * 1 * GMOf((ch.dot(pcvn[v_indexs],sv.T)),1) ##Ec
    obj_pc['pose'] = pprior(0.1)
    obj_pc['betas'] = 0.05*sv.betas
    if regs is not None:
        obj_pc['sph_coll'] = 1e3 * sp
    ch.minimize(
        obj_pc,
        x0=[sv.pose, sv.betas],
        method='dogleg',
        callback=None,
        options={'maxiter': 100,
                 'e_3': .0001,
                 'disp': 0})

    # sv.betas[:] = sv.betas.r + np.hstack((np.array([0.5, 0]), np.zeros(8)))
    # outmesh_path = './fitted_model_%d/' \
    #                'kinect_smpl_%d.obj'%(file_path_number,20)
    # outmesh_path = './fitted_model_%d/' \
    #                'kinect_smpl_%d.obj' % (file_path_number,file_path_number)
    # with open(outmesh_path, 'w') as fp:
    #     for v in sv.r:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #     for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
    #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    _LOGGER.info('end\n')
    t1 = time()
    import sys
    sys.exit()
    # sv.betas[:] = sv.betas.r + np.hstack((np.array([0.5,1]),np.zeros(8)))
    pose_betas_para = {'betas':sv.betas,
                       'pose':sv.pose}
    # pose_betas_para_path = './fitted_model_%d/' \
    #                'pose_betas_parameters.pkl' % (file_path_number)
    pose_betas_para_path = './fitted_model_%d/' \
                           'pose_betas_parameters_%d.pkl' % (file_path_number,file_path_number)
    with open(pose_betas_para_path,'w') as f:
        pickle.dump(pose_betas_para,f)

    # beta_para_path = './fitted_model_%d/' \
    #                  'beta_parameters.pkl' % (file_path_number)
    # with open(beta_para_path,'w') as f:
    #     pickle.dump(sv.betas,f)

    return (sv.pose,sv.betas,sv.trans)

def run_single_fit(file_path_number,
                   img,
                   j3d,
                   hj,
                   pcv,
                   pcvn,
                   conf,
                   model,
                   pyr,
                   regs=None,
                   n_betas=10,
                   flength=5000.,
                   pix_thsh=25.,
                   scale_factor=1,
                   viz=False,
                   do_degrees=None):
    """Run the fit for one specific image.
    :param file_path_number: result save path
    :param img: h x w x 3 image
    :param j2d: 25x3 array of body joints
    :param hj: 38x3 array of hand joints
    :param pc: the vertexs of kinect point cloud
    :param pcvn: the normals of the vertexs of pc
    :param conf: the confidence values
    :param model: SMPL model
    :param pyr: the head pose data
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param scale_factor: int, rescale the image (for LSP, slightly greater images -- 2x -- help obtain better fits)
    :param viz: boolean, if True enables visualization during optimization
    :param do_degrees: list of degrees in azimuth to render the final fit when saving results
    """
    if do_degrees is None:
        do_degrees = []

    # Create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # Get the mean pose as our initial pose
    means=prior.means
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))
    """
    if scale_factor != 1:
        img = cv2.resize(img, (img.shape[1] * scale_factor,
                               img.shape[0] * scale_factor))
        j2d[:, 0] *= scale_factor
        j2d[:, 1] *= scale_factor
    """
    #Estimate initial translation and rotation
    (init_trans,init_rot,ratio) = init_rt(model,j3d,init_pose)

    # Fit
    optimize_on_joints(file_path_number,ratio,model, j3d, hj, pcv, pcvn, init_trans, init_rot, prior, n_betas=n_betas, regs=regs, conf=conf,pyr = pyr)

    # (sv, opt_j2d, t) = optimize_on_joints(
    #     j2d,
    #     model,
    #     cam,
    #     img,
    #     prior,
    #     try_both_orient,
    #     body_orient,
    #     n_betas=n_betas,
    #     conf=conf,
    #     viz=viz,
    #     regs=regs, )



def main(base_dir,
         out_dir,
         use_interpenetration=True,
         n_betas=10,
         flength=5000.,
         pix_thsh=25.,
         use_neutral=False,
         viz=True):
    """Set up paths to image and joint data, saves results.
    :param base_dir: folder containing LSP images and data
    :param out_dir: output folder
    :param use_interpenetration: boolean, if True enables the interpenetration term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (an estimate)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param use_neutral: boolean, if True enables uses the neutral gender SMPL model
    :param viz: boolean, if True enables visualization during optimization
    """
    # with open('./fitted_model_11/tmp.ply','w') as p:
    #     p.write('ply\nformat ascii 1.0\nelement vertex 6890\nproperty float x\nproperty float y\nproperty float z\nelement face 13776\nproperty list uchar int vertex_index\nend_header\n')
    #     with open('./fitted_model_11/kinect_smpl_9.obj','r') as f:
    #         lines = f.readlines()
    #         for e in lines:
    #             s = e.split()
    #             if(s[0] == 'v'):
    #                 p.write('%f %f %f\n'%(float(s[1]),float(s[2]),float(s[3])))
    #             else:
    #                 p.write('3 %d %d %d\n'%(int(s[1])-1,int(s[2])-1,int(s[3])-1))

    img_dir = join(abspath(base_dir), 'images/lsp')
    data_dir = join(abspath(base_dir), 'results/lsp')
    print(data_dir)
    if not exists(out_dir):
        makedirs(out_dir)

    # Render degrees: List of degrees in azimuth to render the final fit.
    # Note that rendering many views can take a while.
    do_degrees = [0.]

    sph_regs = None
    if not use_neutral:
        _LOGGER.info("Reading genders...")
        # File storing information about gender in LSP
        with open(join(data_dir, 'lsp_gender.csv')) as f:
            genders = f.readlines()
        model_female = load_model(MODEL_FEMALE_PATH)
        model_male = load_model(MODEL_MALE_PATH)
        if use_interpenetration:
            sph_regs_male = np.load(SPH_REGS_MALE_PATH)
            sph_regs_female = np.load(SPH_REGS_FEMALE_PATH)
    else:
        gender = 'neutral'
        model = load_model(MODEL_NEUTRAL_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_NEUTRAL_PATH)

    # Load joints
    est = np.load(join(data_dir, 'est_joints.npz'))['est_joints']

    # Read body joints and head direction
    import linecache
    kinect_ids = [18, 17, 16, 12, 13, 14, 10, 9, 8, 4, 5, 6, 2, 3]
    kinect_joints = []
    kinect_conf = []
    pyr = []
    file_path_number = 1001
    # With open("fitted_model_%d/Joints3DPosition_%d.txt"%(file_path_number,file_path_number),"r") as fp:
    with open("fitted_model_%d/Joints3DPosition_%d.txt" % (file_path_number,file_path_number), "r") as fp:
        lines = fp.readlines()
        for l in lines:
            v = l.split()
            if(len(kinect_conf) == 25):
                pyr.append(v[3:])
            else:
                kinect_joints.append(v[2:5])
                kinect_conf.append(v[5])
    kinect_joints = ch.array(kinect_joints)
    kinect_joints = kinect_joints-kinect_joints[1]
    kinect_conf = ch.array(kinect_conf)
    pyr = np.array(pyr[0])

    # Read hand joints
    hj = []
    with open("fitted_model_%d/hand_jointposition_%d.txt" % (file_path_number, file_path_number), "r") as fp:
        lines = fp.readlines()
        for l in lines:
            v = l.split()
            hj.append(v[1:])
    hj = ch.array(hj)

    #Read kinect point cloud
    pcv = [] ##v
    pcvn = [] ##vn
    with open("fitted_model_%d/point_cloud_seg%d.obj" % (file_path_number, file_path_number), "r") as fp:
        while True:
            line_vn = fp.readline()
            if not line_vn:
                break
            line_v = fp.readline()
            vn = line_vn.split()
            pcvn.append(vn[1:]*(-vn[3]/abs(vn[3])))
            pcv.append(line_v.split()[1:])
    pcv = ch.array(pcv)
    pcvn = ch.array(pcvn)

    # Load images
    img_paths = sorted(glob(join(img_dir, '*[0-9].jpg')))
    for ind, img_path in enumerate(img_paths):
        out_path = '%s/%04d.pkl' % (out_dir, ind)
        if True or not exists(out_path):
            _LOGGER.info('Fitting 3D body on `%s` (saving to `%s`).', img_path,
                         out_path)
            img = cv2.imread(img_path)
            if img.ndim == 2:
                _LOGGER.warn("The image is grayscale!")
                img = np.dstack((img, img, img))

            joints = est[:2, :, ind].T
            joints = np.array(kinect_joints)
            conf = est[2, :, ind]
            conf = np.array(kinect_conf)

            if not use_neutral:
                gender = 'male' if int(genders[ind]) == 0 else 'female'
                if gender == 'female':
                    model = model_female
                    if use_interpenetration:
                        sph_regs = sph_regs_female
                elif gender == 'male':
                    model = model_male
                    if use_interpenetration:
                        sph_regs = sph_regs_male

            model  = model_male
            run_single_fit(
                file_path_number,
                img,
                joints,
                hj,
                pcv,
                pcvn,
                +
                --+
                conf,
                model,
                pyr,
                regs=sph_regs,
                n_betas=n_betas,
                flength=flength,
                pix_thsh=pix_thsh,
                scale_factor=2,
                viz=viz,
                do_degrees=do_degrees)

        break

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        'base_dir',
        default='/home/ubuntu/smplify_public/',
        nargs='?',
        help="Directory that contains images/lsp and results/lps , i.e."
        "the directory you untared smplify_code.tar.gz")
    parser.add_argument(
        '--out_dir',
        default='/tmp/smplify_lsp/',
        type=str,
        help='Where results will be saved, default is /tmp/smplify_lsp')
    parser.add_argument(
        '--no_interpenetration',
        default=False,
        action='store_true',
        help="Using this flag removes the interpenetration term, which speeds"
        "up optimization at the expense of possible interpenetration.")
    parser.add_argument(
        '--gender_neutral',
        default=False,
        action='store_true',
        help="Using this flag always uses the neutral SMPL model, otherwise "
        "gender specified SMPL models are used.")
    parser.add_argument(
        '--n_betas',
        default=10,
        type=int,
        help="Specify the number of shape coefficients to use.")
    parser.add_argument(
        '--flength',
        default=5000,
        type=float,
        help="Specify value of focal length.")
    parser.add_argument(
        '--side_view_thsh',
        default=25,
        type=float,
        help="This is thresholding value that determines whether the human is captured in a side view. If the pixel distance between the shoulders is less than this value, two initializations of SMPL fits are tried.")
    parser.add_argument(
        '--viz',
        default=False,
        action='store_true',
        help="Turns on visualization of intermediate optimization steps "
        "and final results.")
    args = parser.parse_args()

    use_interpenetration = not args.no_interpenetration
    if not use_interpenetration:
        _LOGGER.info('Not using interpenetration term.')
    if args.gender_neutral:
        _LOGGER.info('Using gender neutral model.')

    # Set up paths & load models.
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(abspath(dirname(__file__)), 'models')
    # Model paths:
    MODEL_NEUTRAL_PATH = join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(
        MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR,
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    if use_interpenetration:
        # paths to the npz files storing the regressors for capsules
        SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR,
                                     'regressors_locked_normalized_hybrid.npz')
        SPH_REGS_FEMALE_PATH = join(MODEL_DIR,
                                    'regressors_locked_normalized_female.npz')
        SPH_REGS_MALE_PATH = join(MODEL_DIR,
                                  'regressors_locked_normalized_male.npz')

    main(args.base_dir, args.out_dir, use_interpenetration, args.n_betas,
         args.flength, args.side_view_thsh, args.gender_neutral, args.viz)

