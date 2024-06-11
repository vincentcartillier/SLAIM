import os
import sys
import json
import numpy as np
import trimesh
from packaging import version
import open3d as o3d

# -- linking Instant-NGP here
sys.path.append("dependencies/instant-ngp/build/")
import pyngp as ngp # noqa


def nerf_matrix_to_ngp(pose, scale, offset):
    pose[:3,0] *= 1.
    pose[:3,1] *= -1.
    pose[:3,2] *= -1.
    pose[:3,3] = pose[:3,3] * scale + offset

    # cycle axes
    pose[[0,1,2]] = pose[[1,2,0]]
    return pose



#Get the convex hull directly in NGP space
def get_bound_from_frames(poses, instant_ngp,
                          H, W, fx, fy, cx, cy, mesh_bound_scale=1.02, scale=1):
    """
    Get the scene bound (convex hull),
    using sparse estimated camera poses and corresponding depth images.
    return_mesh (trimesh.Trimesh): the convex hull.
    """

    if version.parse(o3d.__version__) >= version.parse('0.13.0'):
        # for new version as provided in environment.yaml
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    else:
        # for lower version
        volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=4.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    cam_points = []
    for e in poses:
        kf_id = e[0]
        c2w = e[1]
        c2w = np.array(c2w)
        if c2w.shape[0] < 4:
            c2w = np.pad(c2w,((0,1), (0,0)))
            c2w[-1,-1] = 1

        w2c = np.linalg.inv(c2w)
        cam_points.append(c2w[:3, 3])

        depth = instant_ngp.get_depth_from_gpu(kf_id)
        depth = np.array(depth)
        depth = depth[:int(H*W)]
        depth = depth.reshape((H,W))

        color = instant_ngp.get_image_from_gpu(kf_id)
        color = np.array(color)
        color = color.reshape((H,W,4))
        color = color.astype(np.uint8)

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(color[:,:,:3])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1,
            depth_trunc=1000,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)

    cam_points = np.stack(cam_points, axis=0)
    mesh = volume.extract_triangle_mesh()
    mesh_points = np.array(mesh.vertices)
    points = np.concatenate([cam_points, mesh_points], axis=0)
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    mesh, _ = o3d_pc.compute_convex_hull()
    mesh.compute_vertex_normals()
    if version.parse(o3d.__version__) >= version.parse('0.13.0'):
        mesh = mesh.scale(mesh_bound_scale, mesh.get_center())
    else:
        mesh = mesh.scale(mesh_bound_scale, center=True)
    points = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    return return_mesh



def get_grid_uniform(resolution):
    x = np.linspace(0., 1., resolution[0])
    y = np.linspace(0., 1., resolution[1])
    z = np.linspace(0., 1., resolution[2])

    gx, gy, gz = np.meshgrid(x, y, z, indexing="xy")

    gx = gx.flatten(); gx = gx[:,np.newaxis]
    gy = gy.flatten(); gy = gy[:,np.newaxis]
    gz = gz.flatten(); gz = gz[:,np.newaxis]

    grid_points = np.concatenate([gx, gy, gz], axis=1)

    return grid_points



def get_convex_hull_mask(instant_ngp, renderer, cfg, res):

    # get DATA
    root = cfg.META.OUTPUT_DIR
    experiment_name = cfg.META.NAME_EXPERIMENT
    run_id = cfg.META.RUN_ID
    run_id = str(run_id)

    experiment_dir = os.path.join(
        root,
        experiment_name,
        run_id,
    )

    poses_dir = os.path.join(
        experiment_dir,
        'poses'
    )

    # - load poses
    filename = os.path.join(
        poses_dir,
        'final_poses_ngp.json'
    )
    pose_filename = filename[:-5]+"_motionBA.json"
    if not os.path.isfile(pose_filename):
        pose_filename = filename[:-5]+"_rec.json"
        if not os.path.isfile(pose_filename):
            pose_filename = filename
            if not os.path.isfile(pose_filename):
                poses_files = os.listdir(poses_dir)
                poses_files = [x for x in poses_files if x[:5]=='poses']
                poses_files_num = [int(x.split('_')[1].split('.')[0]) for x in poses_files]
                assert len(poses_files_num)>0
                poses_max_files_num = np.max(poses_files_num)
                filename = os.path.join(
                    poses_dir,
                    f'poses_{poses_max_files_num}.json'
                )

    all_poses = json.load(open(filename, 'r'))
    all_poses = [np.asarray(x) for x in all_poses]

    num_poses = len(all_poses)

    # - get list of KF with estimated poses
    keyframe_filename=os.path.join(
        poses_dir,
        f'keyframes_final.json'
    )
    if not os.path.isfile(keyframe_filename):
        m = os.listdir(poses_dir)
        m = [x for x in m if x.startswith("keyframes")]
        if len(m) >0:
            m = [x.split(".")[0] for x in m]
            m = [x.split("_")[1] for x in m]
            m = [int(x) for x in m]
            last_model_num = np.max(m)
            keyframe_filename = os.path.join(
                model_dir,
                f"keyframes_{last_model_num}.json"
            )
        else:
            keyframe_filename=None

    if keyframe_filename:
        kf_list = json.load(open(keyframe_filename, "r"))
    else:
        kf_list = list(range(0,num_poses,10))

    assert len(all_poses)>=np.max(kf_list)

    ngp_scale=cfg.RENDERER.SCALE
    ngp_offset=cfg.RENDERER.OFFSET
    kf_poses = [(x, nerf_matrix_to_ngp(all_poses[x], ngp_scale, ngp_offset)) for x in kf_list]
    kf_poses = kf_poses[::10]

    #get convex hull from keyframe list
    cx = renderer.cx
    cy = renderer.cy
    H = renderer.H
    W = renderer.W
    fx = renderer.fx
    fy = renderer.fy

    mesh_convex_hull = get_bound_from_frames(kf_poses, instant_ngp, H, W, fx, fy, cx, cy, mesh_bound_scale=1.02, scale=1)

      output_convex_hull_filename = os.path.join(
        experiment_dir,
        'mesh_convex_hull.obj'
    )
    _ = mesh_convex_hull.export(output_convex_hull_filename)


    # Pre-filtering of points using OBB
    m = mesh_convex_hull.as_open3d
    obb = m.get_oriented_bounding_box()
    points = get_grid_uniform(res)

    mask_outer = np.zeros(points.shape[0], dtype=np.bool)
    bs = 500000
    Ns = len(points) // bs
    if len(points) > bs:
        prev_offset=0
        for i, pnts in enumerate(np.array_split(points, Ns, axis=0)):
            points_indices_within_bb_outer=obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pnts))
            points_indices_within_bb_outer=np.array(points_indices_within_bb_outer)
            points_indices_within_bb_outer+=prev_offset
            points_indices_within_bb_outer=points_indices_within_bb_outer.astype(np.int32)
            mask_outer[points_indices_within_bb_outer] = True
            prev_offset+=pnts.shape[0]
    else:
        points_indices_within_bb_outer=obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
        mask_outer[points_indices_within_bb_outer] = True

    mask = ~mask_outer
    # -- rotate axis
    mask = mask.reshape((res[0], res[1], res[2]))
    mask = np.rollaxis(mask,1)
    mask = mask.flatten()

    return mask
