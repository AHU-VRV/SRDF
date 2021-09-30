
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
from time import time
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

_LOGGER = logging.getLogger(__name__)

# Mapping from LSP joints to SMPL joints.
# 0 Right ankle  8
# 1 Right knee   5
# 2 Right hip    2
# 3 Left hip     1
# 4 Left knee    4
# 5 Left ankle   7
# 6 Right wrist  21
# 7 Right elbow  19
# 8 Right shoulder 17
# 9 Left shoulder  16
# 10 Left elbow    18
# 11 Left wrist    20
# 12 Neck           -
# 13 Head top       added


# --------------------Camera estimation --------------------
def guess_init(model, focal_length, j2d, init_pose):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model: SMPL model
    :param focal_length: camera focal length (kept fixed)
    :param j2d: 14x2 array of CNN joints
    :param init_pose: 72D vector of pose parameters used for initialization (kept fixed)
    :returns: 3D vector corresponding to the estimated camera translation
    """
    cids = np.arange(0, 12)
    # map from LSP to SMPL joints
    j2d_here = j2d[cids]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])
    Jtr = Jtr[smpl_ids].r

    # 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
    diff3d = np.array([Jtr[9] - Jtr[3], Jtr[8] - Jtr[2]])
    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

    diff2d = np.array([j2d_here[9] - j2d_here[3], j2d_here[8] - j2d_here[2]])
    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

    est_d = focal_length * (mean_height3d / mean_height2d)
    # just set the z value
    init_t = np.array([0., 0., est_d])
    return init_t


def initialize_camera(model,
                      j2d,
                      img,
                      init_pose,
                      flength=5000.,
                      pix_thsh=25.,
                      viz=False):
    """Initialize camera translation and body orientation
    :param model: SMPL model
    :param j2d: 14x2 array of CNN joints
    :param img: h x w x 3 image 
    :param init_pose: 72D vector of pose parameters used for initialization
    :param flength: camera focal length (kept fixed)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing the estimated camera,
              a boolean deciding if both the optimized body orientation and its flip should be considered,
              3D vector for the body orientation
    """
    # optimize camera translation and body orientation based on torso joints
    # LSP torso ids:
    # 2=right hip, 3=left hip, 8=right shoulder, 9=left shoulder
    torso_cids = [2, 3, 8, 9]
    # corresponding SMPL torso ids
    torso_smpl_ids = [2, 1, 17, 16]

    center = np.array([img.shape[1] / 2, img.shape[0] / 2])

    # initialize camera rotation
    rt = ch.zeros(3)
    # initialize camera translation
    _LOGGER.info('initializing translation via similar triangles')
    init_t = guess_init(model, flength, j2d, init_pose)
    t = ch.array(init_t)

    # check how close the shoulder joints are
    try_both_orient = np.linalg.norm(j2d[8] - j2d[9]) < pix_thsh

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])

    # initialize the camera
    cam = ProjectPoints(
        f=np.array([flength, flength]), rt=rt, t=t, k=np.zeros(5), c=center)

    # we are going to project the SMPL joints
    cam.v = Jtr

    if viz:
        viz_img = img.copy()

        # draw the target (CNN) joints
        for coord in np.around(j2d).astype(int):
            if (coord[0] < img.shape[1] and coord[0] >= 0 and
                    coord[1] < img.shape[0] and coord[1] >= 0):
                cv2.circle(viz_img, tuple(coord), 3, [0, 255, 0])

        import matplotlib.pyplot as plt
        plt.ion()

        # draw optimized joints at each iteration
        def on_step(_):
            """Draw a visualization."""
            plt.figure(1, figsize=(5, 5))
            plt.subplot(1, 1, 1)
            viz_img = img.copy()
            for coord in np.around(cam.r[torso_smpl_ids]).astype(int):
                if (coord[0] < viz_img.shape[1] and coord[0] >= 0 and
                        coord[1] < viz_img.shape[0] and coord[1] >= 0):
                    cv2.circle(viz_img, tuple(coord), 3, [0, 0, 255])
            plt.imshow(viz_img[:, :, ::-1])
            plt.draw()
            plt.show()
            plt.pause(1e-3)
    else:
        on_step = None
    # optimize for camera translation and body orientation
    free_variables = [cam.t, opt_pose[:3]]
    ch.minimize(
        # data term defined over torso joints...
        {'cam': j2d[torso_cids] - cam[torso_smpl_ids],
         # ...plus a regularizer for the camera translation
         'cam_t': 1e2 * (cam.t[2] - init_t[2])},
        x0=free_variables,
        method='dogleg',
        callback=on_step,
        options={'maxiter': 100,
                 'e_3': .0001,
                 # disp set to 1 enables verbose output from the optimizer
                 'disp': 0})
    if viz:
        plt.ioff()
    return (cam, try_both_orient, opt_pose[:3].r)


# # --------------------Core optimization --------------------
# def optimize_on_joints(j2d,
#                        model,
#                        cam,
#                        img,
#                        prior,
#                        try_both_orient,
#                        body_orient,
#                        n_betas=10,
#                        regs=None,
#                        conf=None,
#                        viz=False):
#     """Fit the model to the given set of joints, given the estimated camera
#     :param j2d: 14x2 array of CNN joints
#     :param model: SMPL model
#     :param cam: estimated camera
#     :param img: h x w x 3 image
#     :param prior: mixture of gaussians pose prior
#     :param try_both_orient: boolean, if True both body_orient and its flip are considered for the fit
#     :param body_orient: 3D vector, initialization for the body orientation
#     :param n_betas: number of shape coefficients considered during optimization
#     :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
#     :param conf: 14D vector storing the confidence values from the CNN
#     :param viz: boolean, if True enables visualization during optimization
#     :returns: a tuple containing the optimized model, its joints projected on image space, the camera translation
#     """
#     t0 = time()
#     # define the mapping LSP joints -> SMPL joints
#     # cids are joints ids for LSP:
#     cids = range(12) + [13]
#     # joint ids for SMPL
#     # SMPL does not have a joint for head, instead we use a vertex for the head
#     # and append it later.
#     smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]
#
#     # the vertex id for the joint corresponding to the head
#     head_id = 411
#
#     # weights assigned to each joint during optimization;
#     # the definition of hips in SMPL and LSP is significantly different so set
#     # their weights to zero
#     base_weights = np.array(
#         [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
#
#     if try_both_orient:
#         flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
#             cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
#         flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
#         orientations = [body_orient, flipped_orient]
#     else:
#         orientations = [body_orient]
#
#     if try_both_orient:
#         # store here the final error for both orientations,
#         # and pick the orientation resulting in the lowest error
#         errors = []
#
#     svs = []
#     cams = []
#     for o_id, orient in enumerate(orientations):
#         # initialize the shape to the mean shape in the SMPL training set
#         betas = ch.zeros(n_betas)
#
#         # initialize the pose by using the optimized body orientation and the
#         # pose prior
#         init_pose = np.hstack((orient, prior.weights.dot(prior.means)))
#
#         # instantiate the model:
#         # verts_decorated allows us to define how many
#         # shape coefficients (directions) we want to consider (here, n_betas)
#         sv = verts_decorated(
#             trans=ch.zeros(3),
#             pose=ch.array(init_pose),
#             v_template=model.v_template,
#             J=model.J_regressor,
#             betas=betas,
#             shapedirs=model.shapedirs[:, :, :n_betas],
#             weights=model.weights,
#             kintree_table=model.kintree_table,
#             bs_style=model.bs_style,
#             f=model.f,
#             bs_type=model.bs_type,
#             posedirs=model.posedirs)
#
#         test = prior(sv.pose)
#
#         outmesh_path = './hello_smpl.obj'
#         with open(outmesh_path, 'w') as fp:
#             for v in sv.r:
#                 fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
#             for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
#                 fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
#
#         # make the SMPL joints depend on betas
#         Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
#                            for i in range(len(betas))])
#         J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
#             model.v_template.r)
#
#         # get joint positions as a function of model pose, betas and trans
#         (_, A_global) = global_rigid_transformation(
#             sv.pose, J_onbetas, model.kintree_table, xp=ch)
#         Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans
#
#         # add the head joint, corresponding to a vertex...
#         Jtr = ch.vstack((Jtr, sv[head_id]))
#
#         # ... and add the joint id to the list
#         if o_id == 0:
#             smpl_ids.append(len(Jtr) - 1)
#
#         # update the weights using confidence values
#         weights = base_weights * conf[
#             cids] if conf is not None else base_weights
#
#         # project SMPL joints on the image plane using the estimated camera
#         cam.v = Jtr
#
#         # data term: distance between observed and estimated joints in 2D
#         ##EJ
#         obj_j2d = lambda w, sigma: (
#             w * weights.reshape((-1, 1)) * GMOf((j2d[cids] - cam[smpl_ids]), sigma))
#
#         # mixture of gaussians pose prior
#         pprior = lambda w: w * prior(sv.pose)
#         # joint angles pose prior, defined over a subset of pose parameters:
#         # 55: left elbow,  90deg bend at -np.pi/2
#         # 58: right elbow, 90deg bend at np.pi/2
#         # 12: left knee,   90deg bend at np.pi/2
#         # 15: right knee,  90deg bend at np.pi/2
#         alpha = 10
#         my_exp = lambda x: alpha * ch.exp(x)
#         ##Ea
#         obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
#                                                  58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])
#
#         if viz:
#             import matplotlib.pyplot as plt
#             plt.ion()
#
#             def on_step(_):
#                 """Create visualization."""
#                 plt.figure(1, figsize=(10, 10))
#                 plt.subplot(1, 2, 1)
#                 # show optimized joints in 2D
#                 tmp_img = img.copy()
#                 for coord, target_coord in zip(
#                         np.around(cam.r[smpl_ids]).astype(int),
#                         np.around(j2d[cids]).astype(int)):
#                     if (coord[0] < tmp_img.shape[1] and coord[0] >= 0 and
#                             coord[1] < tmp_img.shape[0] and coord[1] >= 0):
#                         cv2.circle(tmp_img, tuple(coord), 3, [0, 0, 255])
#                     if (target_coord[0] < tmp_img.shape[1] and
#                             target_coord[0] >= 0 and
#                             target_coord[1] < tmp_img.shape[0] and
#                             target_coord[1] >= 0):
#                         cv2.circle(tmp_img, tuple(target_coord), 3,
#                                    [0, 255, 0])
#                 plt.imshow(tmp_img[:, :, ::-1])
#                 plt.draw()
#                 plt.show()
#                 plt.pause(1e-2)
#
#             on_step(_)
#         else:
#             on_step = None
#
#         if regs is not None:
#             # interpenetration term
#             ##Esp
#             sp = SphereCollisions(
#                 pose=sv.pose, betas=sv.betas, model=model, regs=regs)
#             sp.no_hands = True
#         # weight configuration used in the paper, with joints + confidence values from the CNN
#         # (all the weights used in the code were obtained via grid search, see the paper for more details)
#         # the first list contains the weights for the pose priors,
#         # the second list contains the weights for the shape prior
#         opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
#                           [1e2, 5 * 1e1, 1e1, .5 * 1e1])
#
#         # run the optimization in 4 stages, progressively decreasing the
#         # weights for the priors
#         for stage, (w, wbetas) in enumerate(opt_weights):
#             _LOGGER.info('stage %01d', stage)
#             objs = {}
#
#             objs['j2d'] = obj_j2d(1., 100)##EJ
#
#             objs['pose'] = pprior(w)##Ea
#
#             objs['pose_exp'] = obj_angle(0.317 * w)##Es
#
#             objs['betas'] = wbetas * betas
#
#             if regs is not None:
#                 objs['sph_coll'] = 1e3 * sp##Esp
#
#             ch.minimize(
#                 objs,
#                 x0=[sv.betas, sv.pose],
#                 method='dogleg',
#                 callback=on_step,
#                 options={'maxiter': 100,
#                          'e_3': .0001,
#                          'disp': 0})
#
#         outmesh_path = './hello_smpl.obj'
#         with open(outmesh_path, 'w') as fp:
#             mm = sv.r
#             for v in sv.r:
#                 fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
#             nn = sv.f
#             nn1 = nn + 1
#             for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
#                 fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
#
#         t1 = time()
#         _LOGGER.info('elapsed %.05f', (t1 - t0))
#         if try_both_orient:
#             errors.append((objs['j2d'].r**2).sum())
#         svs.append(sv)
#         cams.append(cam)
#
#     if try_both_orient and errors[0] > errors[1]:
#         choose_id = 1
#     else:
#         choose_id = 0
#     if viz:
#         plt.ioff()
#     return (svs[choose_id], cams[choose_id].r, cams[choose_id].t.r)

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

#init translation and rotation
#use four joints: Spine mid, Shoulder left, Shoulder right, Spine Shoulder.
def init_rt(model,j2d,init_pose):

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

    ratio = np.linalg.norm(j2d[4]-j2d[8])/np.linalg.norm(Jtr[16]-Jtr[17])
    ratio = 0.9
    a = np.linalg.norm(j2d[12]-j2d[16])/np.linalg.norm(Jtr[1]-Jtr[2])
    b = np.linalg.norm(j2d[4]-j2d[8])/np.linalg.norm(Jtr[16]-Jtr[17])


    ch.minimize(
        {'rt':j2d[kin_ids]-(Jtr[smpl_ids]*ratio+init_trans)},
        x0=[opt_pose[:3],init_trans],
        method='dogleg',
        callback=None,
        options={'maxiter':100,
                 'e_3':.0001,
                 'disp':0})

    Jtr0 = Jtr*ratio

    # init_trans = (j2d[4]+j2d[8])/2-(Jtr[16]*ratio+Jtr[17]*ratio)/2
    return (init_trans,opt_pose[:3].r,ratio)

def optimize_on_joints(ratio,model,j2d,init_trans,init_rot,prior,n_betas=10,regs=None,conf=None,pyr = None):
    t0=time()

    #define the mapping: kinect joints to smpl joints
    #SMPL does not have a joint for head, instead we use a vertex for the head
    # and append it later.
    kinect_ids = [4,5,6,8,9,10,13,14,17,18,20,2]
    smpl_ids = [16,18,20,17,19,21,4,7,5,8,12,15]


    # the vertex id for the joint corresponding to the head
    head_id = 411

    # weights assigned to each joint during optimization;
    base_weights = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,0.95], dtype=np.float64)

    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(n_betas)

    # initialize the pose by using the initial rotation and the
    # pose prior
    init_rot = np.zeros(3)
    init_pose = np.hstack((init_rot, prior.weights.dot(prior.means)))
   # init_pose = np.hstack((init_rot,np.zeros(69)))
    init_trans = ch.array([0,0.2,0])
    # instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)
    init_trans = ch.array([0,0,0])
    sv = verts_decorated(
        trans=init_trans,
        pose=ch.array(init_pose),
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

    # make the SMPL joints depend on betas
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                       for i in range(len(betas))])
    J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
        model.v_template.r)
    # init_trans = ch.zeros(3)
    # get joint positions as a function of model pose, betas and trans
    (_, A_global) = global_rigid_transformation(
        sv.pose, J_onbetas, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3]+init_trans for g in A_global])
    # Jtr = Jtr + init_trans #+ ch.array([0,0.17,-0.13])
    # add the head joint, corresponding to a vertex...
    #Jtr = ch.vstack((Jtr, sv[head_id])) + sv.trans

    # ... and add the joint id to the list
    #smpl_ids.append(len(Jtr) - 1)

    # update the weights using confidence values
    weights = base_weights * conf[kinect_ids] if conf is not None else base_weights

    # data term: distance between observed and estimated joints in 2D
    ##EJ
    obj_j2d = lambda w, sigma: (w * weights.reshape((-1, 1)) * GMOf((j2d[kinect_ids] - sv.J_transformed[smpl_ids]), sigma))

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

    outmesh_path = './kinect_smpl_pre0.obj'
    sv1 = sv+init_trans
    with open(outmesh_path, 'w') as fp:
        for v in sv.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    outmesh_path = './kinect_smpl_pre1.obj'

    with open(outmesh_path, 'w') as fp:
        for v in sv1.r:
            fp.write('v %f %f %f\n' % (v[0], v[1]+0.15, v[2]-0.13))
        for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    # kinect_ids = [13,17,1,14,18,15,19,20,2,4,8,5,9,6,10,7,11]
    # smpl_ids = [4,5,6,7,8,10,11,12,15,16,17,18,19,20,21,22,23]
    # kinect_ids = [13,17,5,6,9,10]
    # smpl_ids = [4,5,18,20,19,21]

    # for ids in range(19):
    #     sid0 = sv.kintree_table[0, smpl_ids[ids]] * 3
    #     sid1 = sid0 + 3
    #     if(sid0 == 0):
    #         continue
    #
    #     objs = {}
    #
    #     objs['j2d'] = 1.0 * weights.reshape((-1, 1)) * GMOf((j2d[kinect_ids[ids]] - Jtr[smpl_ids[ids]]), 100)  ##EJ
    #
    #     objs['pose'] = init_pose  ##Ea
    #
    #     # objs['pose_exp'] = obj_angle(0.317 * w)  ##Es
    #
    #     objs['betas'] = betas
    #
    #     if regs is not None:
    #         objs['sph_coll'] = 1e3 * sp  ##Esp
    #
    #     ch.minimize(
    #         objs,
    #         x0=[sv.betas, sv.pose[sid0:sid1]],
    #         method='dogleg',
    #         callback=None,
    #         options={'maxiter': 100,
    #                  'e_3': .0001,
    #                  'disp': 0})
    #
    #     outmesh_path = './kinect_smpl.obj'
    #     with open(outmesh_path, 'w') as fp:
    #         for v in sv.r:
    #             fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #         for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
    #             fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    w0 = 1
    w1 = 1
    w2 = 0.1
    #bei fen
    for ids in range(10):
        objs = {}

        if (ids == 0 or ids == 9):
            objs['betas'] = w2 * betas
        else:
            tmpBetas = ch.zeros(10)
            tmpBetas[:] = betas.r
            objs['betas'] = betas - tmpBetas

        objs['j2d'] = obj_j2d(w0, 100)
        w0 += 1

        objs['pose'] = pprior(w1)
        w1 -= 0.1
        objs['trans'] = 10 * init_trans

        if(ids == 8):
            w0 = 1
            w1 = 0.1

        if regs is not None:
            objs['sph_coll'] = 1e3 * sp  ##Esp
        # if(ids == 0):
        #     objs['j2d'] = obj_j2d(w0, 100)
        #     objs['trans'] = 10 * init_trans
        #     objs['betas'] = w2 * betas
        #     if regs is not None:
        #         objs['sph_coll'] = 1e3 * sp  ##Esp
        #     free_variables = [sv.betas,sv.trans]
        # else:
        #     tmpBetas = ch.zeros(10)
        #     tmpBetas[:] = betas.r
        #     objs['j2d'] = obj_j2d(w0, 100)
        #     w0 += 1
        #     #objs['pose1'] =  init_pose  ##Ea
        #     objs['pose'] = pprior(w1)
        #     #objs['pose2'] = (w1 / 10) * init_pose[3:]
        #     objs['trans'] = 10 * init_trans
        #     objs['betas'] =  betas-tmpBetas
        #     w1 -= 0.1
        #     if regs is not None:
        #         objs['sph_coll'] = 1e3 * sp  ##Esp
        free_variables = [sv.betas,sv.pose,sv.trans]
        ch.minimize(
            objs,
            x0=free_variables,
            method='dogleg',
            callback=None,
            options={'maxiter': 100,
                     'e_3': .0001,
                     'disp': 0})

        outmesh_path = '%s%d.obj'%('./fitted_model_9/kinect_smpl_',ids)
        with open(outmesh_path, 'w') as fp:
            for v in sv.r:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in sv.f+1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    sv.pose[45:48] = pyr
    outmesh_path = '%s%d.obj' % ('./fitted_model_9/kinect_smpl_', 10)
    with open(outmesh_path, 'w') as fp:
        for v in sv.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    # for stage, (w, wbetas) in enumerate(opt_weights):
    #     _LOGGER.info('stage %01d', stage)
    #     objs = {}
    #
    #     objs['j2d'] = obj_j2d(1., 100)  ##EJ
    #
    #     objs['pose'] = init_pose ##Ea
    #
    #     #objs['pose_exp'] = obj_angle(0.317 * w)  ##Es
    #
    #     objs['betas'] = wbetas * betas
    #
    #     if regs is not None:
    #         objs['sph_coll'] = 1 * sp  ##Esp
    #
    #     ch.minimize(
    #         objs,
    #         x0=[sv.betas, sv.pose[48:51]],
    #         method='dogleg',
    #         callback=None,
    #         options={'maxiter': 100,
    #                  'e_3': .0001,
    #                  'disp': 0})
    #
    #     outmesh_path = './kinect_smpl_%d.obj'+(ids)
    #     with open(outmesh_path, 'w') as fp:
    #         for v in sv.r:
    #             fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #         for f in sv.f + 1:  # Faces are 1-based, not 0-based in obj files
    #             fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    t1 = time()



    return (sv.pose,sv.betas,sv.trans)

def run_single_fit(img,
                   j2d,
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
    :param img: h x w x 3 image 
    :param j2d: 14x2 array of CNN joints
    :param conf: 14D vector storing the confidence values from the CNN
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param scale_factor: int, rescale the image (for LSP, slightly greater images -- 2x -- help obtain better fits)
    :param viz: boolean, if True enables visualization during optimization
    :param do_degrees: list of degrees in azimuth to render the final fit when saving results
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """
    if do_degrees is None:
        do_degrees = []

    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    means=prior.means
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))
    """
    if scale_factor != 1:
        img = cv2.resize(img, (img.shape[1] * scale_factor,
                               img.shape[0] * scale_factor))
        j2d[:, 0] *= scale_factor
        j2d[:, 1] *= scale_factor
    """
    #estimate initial translation and rotation
    (init_trans,init_rot,ratio) = init_rt(model,j2d,init_pose)

    # estimate the camera parameters
    # (cam, try_both_orient, body_orient) = initialize_camera(
    #     model,
    #     j2d,
    #     img,
    #     init_pose,
    #     flength=flength,
    #     pix_thsh=pix_thsh,
    #     viz=viz)

    # fit
    optimize_on_joints(ratio,model, j2d, init_trans, init_rot, prior, n_betas=n_betas, regs=regs, conf=conf,pyr = pyr)

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

    ########################################
    import linecache
    kinect_ids = [18, 17, 16, 12, 13, 14, 10, 9, 8, 4, 5, 6, 2, 3]
    kinect_joints = []
    kinect_conf = []
    pyr = []
    #joints = kinect_joints[kinect_ids]
    with open("fitted_model_9/Joints3DPosition_9.txt","r") as fp:
        lines = fp.readlines()
        for l in lines:
            v = l.split()
            kinect_joints.append(v[2:5])
            kinect_conf.append(v[5])
    kinect_joints = ch.array(kinect_joints)
    kinect_joints = kinect_joints-kinect_joints[1]
    kinect_conf = ch.array(kinect_conf)

    ########################################

    # Load images
    img_paths = sorted(glob(join(img_dir, '*[0-9].jpg')))
    for ind, img_path in enumerate(img_paths):
        out_path = '%s/%04d.pkl' % (out_dir, ind)
        if not exists(out_path):
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
                img,
                joints,
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
