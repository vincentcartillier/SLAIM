import os
import numpy as np
import open3d as o3d
import random
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from multiprocessing import Pool

from .build import EVALUATOR_REGISTRY

__all__ = ["Evaluator3DReconstruction"]


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(float))
    return comp_ratio


def accuracy(gt_points, rec_points, debug_dir=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)

    #DEBUG
    #DEBUG
    if debug_dir is not None:
        mask = distances > 0.1
        pc = rec_points[mask,:]
        rgb = np.zeros(pc.shape)
        rgb[:,0] = 1.0
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(pc)
        tmp_pcd.colors = o3d.utility.Vector3dVector(rgb)
        filename=os.path.join(debug_dir,"debug_acc_pc.ply")
        o3d.io.write_point_cloud(filename, tmp_pcd)
    #DEBUG
    #DEBUG

    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points, debug_dir):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)

    #DEBUG
    #DEBUG
    if debug_dir is not None:
        mask = distances > 0.1
        pc = gt_points[mask,:]
        rgb = np.zeros(pc.shape)
        rgb[:,1] = 1.0
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(pc)
        tmp_pcd.colors = o3d.utility.Vector3dVector(rgb)
        filename=os.path.join(debug_dir,"debug_comp_pc.ply")
        o3d.io.write_point_cloud(filename, tmp_pcd)
    #DEBUG
    #DEBUG

    comp = np.mean(distances)
    return comp


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc,
        o3d_gt_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c@homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:]+1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True, debug_dir=None):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices, debug_dir)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices, debug_dir)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    print('accuracy: ', accuracy_rec)
    print('completion: ', completion_rec)
    print('completion ratio: ', completion_ratio_rec)

    return {
        'acc': accuracy_rec,
        'completion': completion_rec,
        'completion_ration': completion_ratio_rec
    }



def get_cam_position(gt_meshfile, sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    mesh_gt = trimesh.load(gt_meshfile)
    # Tbw: world_to_bound, bound is defined at the centre of cuboid
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= sz
    extents[1] *= sy
    extents[0] *= sx
    # Twb: bound_to_world
    transform = np.linalg.inv(to_origin)
    transform[0, 3] += dx
    transform[1, 3] += dy
    transform[2, 3] += dz
    return extents, transform



def calc_2d_metric_old(rec_meshfile, gt_meshfile, align=True, n_imgs=1000):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    unseen_gt_pointcloud_file = gt_meshfile.replace('.ply', '_pc_unseen.npy')
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in tqdm(range(n_imgs)):
        while True:
            # sample view, and check if unseen region is not inside the camera view
            # if inside, then needs to resample
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            target = [tx, ty, tz]
            target = np.array(target)-np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if (~seen):
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param,True)

        vis.add_geometry(gt_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param,True)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = vis.capture_depth_float_buffer(False)
        gt_depth = np.asarray(gt_depth)
        vis.remove_geometry(gt_mesh, reset_bounding_box=True,)

        vis.add_geometry(rec_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param,True)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(False)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(rec_mesh, reset_bounding_box=True,)

        errors += [np.abs(gt_depth-ours_depth).mean()]

    errors = np.array(errors)
    # from m to cm
    print('Depth L1(cm): ', errors.mean()*100)
    return {
        "depth-L1": errors.mean()*100
    }



def calc_2d_metric(rec_meshfile, gt_meshfile, unseen_gt_pcd_file,
                   pose_file=None, gt_depth_render_file=None,
                   depth_render_file=None, suffix="virt_cams",
                   align=True, n_imgs=1000, not_counting_missing_depth=True,
                   sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    pc_unseen = np.load(unseen_gt_pcd_file)

    if pose_file and os.path.exists(pose_file):
        sampled_poses = np.load(pose_file)["poses"]
        assert len(sampled_poses) == n_imgs
        print("Found saved renering poses! Loading from disk!!!")
    else:
        sampled_poses = None
        print("Saved renering poses NOT FOUND! Will do the sampling")
    if gt_depth_render_file and os.path.exists(gt_depth_render_file):
        gt_depth_renderings = np.load(gt_depth_render_file)["depths"]
        assert len(gt_depth_renderings) == n_imgs
        print("Found saved renered gt depths! Loading from disk!!!")
    else:
        gt_depth_renderings = None
        print("Saved renered gt depths NOT FOUND! Will re-render!!!")
    if depth_render_file and os.path.exists(depth_render_file):
        depth_renderings = np.load(depth_render_file)["depths"]
        assert len(depth_renderings) == n_imgs
        print("Found saved renered reconstructed depth! Loading from disk!!!")
    else:
        depth_renderings = None
        print("Saved renered reconstructed depth NOT FOUND! Will re-render!!!")

    gt_dir = os.path.dirname(unseen_gt_pcd_file)
    log_dir = os.path.dirname(rec_meshfile)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile, sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    poses = []
    gt_depths = []
    depths = []
    for i in tqdm(range(n_imgs)):
        if sampled_poses is None:
            while True:
                # sample view, and check if unseen region is not inside the camera view
                # if inside, then needs to resample
                up = [0, 0, -1]
                origin = trimesh.sample.volume_rectangular(extents, 1, transform=transform)
                origin = origin.reshape(-1)
                tx = round(random.uniform(-10000, +10000), 2)
                ty = round(random.uniform(-10000, +10000), 2)
                tz = round(random.uniform(-10000, +10000), 2)
                target = [tx, ty, tz]
                target = np.array(target)-np.array(origin)
                c2w = viewmatrix(target, up, origin)
                tmp = np.eye(4)
                tmp[:3, :] = c2w
                c2w = tmp
                seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
                if (~seen):
                    break
            poses.append(c2w)
        else:
            c2w = sampled_poses[i]

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param,True)

        if gt_depth_renderings is None:
            vis.add_geometry(gt_mesh, reset_bounding_box=True,)
            ctr.convert_from_pinhole_camera_parameters(param,True)
            vis.poll_events()
            vis.update_renderer()
            gt_depth = vis.capture_depth_float_buffer(False)
            gt_depth = np.asarray(gt_depth)
            vis.remove_geometry(gt_mesh, reset_bounding_box=True,)
            gt_depths.append(gt_depth)
        else:
            gt_depth = gt_depth_renderings[i]

        if depth_renderings is None:
            vis.add_geometry(rec_mesh, reset_bounding_box=True,)
            ctr.convert_from_pinhole_camera_parameters(param,True)
            vis.poll_events()
            vis.update_renderer()
            ours_depth = vis.capture_depth_float_buffer(False)
            ours_depth = np.asarray(ours_depth)
            vis.remove_geometry(rec_mesh, reset_bounding_box=True,)
            depths.append(ours_depth)
        else:
            ours_depth = depth_renderings[i]


        if not_counting_missing_depth:
            valid_mask = (gt_depth > 0.) & (gt_depth < 19.)
            if np.count_nonzero(valid_mask) <= 100:
                continue
            errors += [np.abs(gt_depth[valid_mask] - ours_depth[valid_mask]).mean()]
        else:
            errors += [np.abs(gt_depth-ours_depth).mean()]

    if pose_file is None:
        np.savez(os.path.join(gt_dir, "sampled_poses_{}.npz".format(n_imgs)), poses=poses)
    elif not os.path.exists(pose_file):
        np.savez(pose_file, poses=poses)

    if gt_depth_render_file is None:
        np.savez(os.path.join(gt_dir, "gt_depths_{}.npz".format(n_imgs)), depths=gt_depths)
    elif not os.path.exists(gt_depth_render_file):
        np.savez(gt_depth_render_file, depths=gt_depths)

    if depth_render_file is None:
        np.savez(os.path.join(log_dir, "depths_{}_{}.npz".format(suffix, n_imgs)), depths=depths)
    elif not os.path.exists(depth_render_file):
        np.savez(depth_render_file, depths=depths)


    errors = np.array(errors)
    # from m to cm
    print('Depth L1(cm): ', errors.mean()*100)
    return {
        "depth-L1": errors.mean()*100
    }




def render_depth(inputs):
    i = inputs["input"]
    render_gt = inputs['render_gt']
    render_pred = inputs['render_pred']
    if render_gt:
        gt_vertices = inputs["gt_vertices"]
        gt_triangles = inputs["gt_triangles"]
        gt_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(gt_vertices),
            o3d.utility.Vector3iVector(gt_triangles)
        )
    if render_pred:
        pred_vertices = inputs["pred_vertices"]
        pred_triangles = inputs["pred_triangles"]
        pred_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(pred_vertices),
            o3d.utility.Vector3iVector(pred_triangles)
        )

    poses = inputs["poses"]
    H = inputs["H"]
    W = inputs["W"]
    K = inputs["K"]
    far = inputs["far"]

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(far)

    gt_depth_maps = []
    pred_depth_maps = []
    for c2w in poses:

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0,0], K[1,1], K[0,2], K[1,2])

        ctr.convert_from_pinhole_camera_parameters(param,True)

        if render_gt:
            vis.add_geometry(gt_mesh, reset_bounding_box=True,)
            ctr.convert_from_pinhole_camera_parameters(param,True)
            vis.poll_events()
            vis.update_renderer()
            gt_depth = vis.capture_depth_float_buffer(False)
            gt_depth = np.asarray(gt_depth)
            vis.remove_geometry(gt_mesh, reset_bounding_box=True,)
            gt_depth_maps.append(gt_depth)

        if render_pred:
            vis.add_geometry(pred_mesh, reset_bounding_box=True,)
            ctr.convert_from_pinhole_camera_parameters(param,True)
            vis.poll_events()
            vis.update_renderer()
            ours_depth = vis.capture_depth_float_buffer(False)
            ours_depth = np.asarray(ours_depth)
            vis.remove_geometry(pred_mesh, reset_bounding_box=True,)
            pred_depth_maps.append(ours_depth)


    return {"index":i, "gt_depths": gt_depth_maps, "pred_depths": pred_depth_maps}






def calc_2d_metric_parallel(rec_meshfile, gt_meshfile, unseen_gt_pcd_file,
                   pose_file=None, gt_depth_render_file=None,
                   depth_render_file=None, suffix="virt_cams",
                   align=True, n_imgs=1000, not_counting_missing_depth=True,
                   sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5
    K = np.array(
        [
            [fx, 0,cx],
            [ 0,fy,cy],
            [ 0, 0, 1],
        ]
    )
    far=20.


    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    if os.path.isfile(unseen_gt_pcd_file):
        pc_unseen = np.load(unseen_gt_pcd_file)
    else:
        pc_unseen=None

    if pose_file and os.path.exists(pose_file):
        sampled_poses = np.load(pose_file)["poses"]
        assert len(sampled_poses) == n_imgs
        print("Found saved renering poses! Loading from disk!!!")
    else:
        sampled_poses = None
        print("Saved renering poses NOT FOUND! Will do the sampling")
    if gt_depth_render_file and os.path.exists(gt_depth_render_file):
        gt_depth_renderings = np.load(gt_depth_render_file)["depths"]
        assert len(gt_depth_renderings) == n_imgs
        print("Found saved renered gt depths! Loading from disk!!!")
    else:
        gt_depth_renderings = None
        print("Saved renered gt depths NOT FOUND! Will re-render!!!")
    if depth_render_file and os.path.exists(depth_render_file):
        depth_renderings = np.load(depth_render_file)["depths"]
        assert len(depth_renderings) == n_imgs
        print("Found saved renered reconstructed depth! Loading from disk!!!")
    else:
        depth_renderings = None
        print("Saved renered reconstructed depth NOT FOUND! Will re-render!!!")

    if os.path.isfile(unseen_gt_pcd_file):
        gt_dir = os.path.dirname(unseen_gt_pcd_file)
    else:
        gt_dir = os.path.dirname(gt_meshfile)
    log_dir = os.path.dirname(rec_meshfile)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile, sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz)

    errors = []
    poses = []
    gt_depths = []
    depths = []

    if (gt_depth_renderings is None) or (depth_renderings is None):
        # get the poses
        if sampled_poses is None:
            for i in range(n_imgs):
                while True:
                    # sample view, and check if unseen region is not inside the camera view
                    # if inside, then needs to resample
                    up = [0, 0, -1]
                    origin = trimesh.sample.volume_rectangular(extents, 1, transform=transform)
                    origin = origin.reshape(-1)
                    tx = round(random.uniform(-10000, +10000), 2)
                    ty = round(random.uniform(-10000, +10000), 2)
                    tz = round(random.uniform(-10000, +10000), 2)
                    target = [tx, ty, tz]
                    target = np.array(target)-np.array(origin)
                    c2w = viewmatrix(target, up, origin)
                    tmp = np.eye(4)
                    tmp[:3, :] = c2w
                    c2w = tmp
                    seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
                    if (~seen):
                        break
                poses.append(c2w)
        else:
            poses = sampled_poses

        gt_vertices = np.array(gt_mesh.vertices)
        gt_triangles = np.array(gt_mesh.triangles)

        pred_vertices = np.array(rec_mesh.vertices)
        pred_triangles = np.array(rec_mesh.triangles)

        BS = int(n_imgs/8) + 1
        inputs = [{'input':i,
                   'gt_vertices': gt_vertices,
                   'gt_triangles': gt_triangles,
                   'pred_vertices': pred_vertices,
                   'pred_triangles': pred_triangles,
                   'render_gt': gt_depth_renderings is None,
                   'render_pred': depth_renderings is None,
                   'H': H,
                   'W': W,
                   'K': K,
                   'far': far,
                   'poses': np.array(poses[i:i+BS]),
                  } for i in range(0,len(poses),BS)]

        pool = Pool(8)

        results = list(
            tqdm(
                pool.imap_unordered(
                    render_depth, inputs),
                    total=len(inputs)
            )
        )

        results = sorted(results, key=lambda d: d["index"])

        if gt_depth_renderings is None:
            for r in results:
                for d in r["gt_depths"]:
                    gt_depths.append(d)
        else:
            gt_depths = gt_depth_renderings

        if depth_renderings is None:
            for r in results:
                for d in r["pred_depths"]:
                    depths.append(d)
        else:
            depths = depth_renderings

        assert len(depths) == len(poses)
        assert len(gt_depths) == len(poses)

    else:
        gt_depths = gt_depth_renderings
        depths = depth_renderings
        #No need to re-render depths

    for i in range(n_imgs):
        gt_depth = gt_depths[i]
        ours_depth = depths[i]
        if not_counting_missing_depth:
            valid_mask = (gt_depth > 0.) & (gt_depth < 19.)
            if np.count_nonzero(valid_mask) <= 100:
                continue
            errors += [np.abs(gt_depth[valid_mask] - ours_depth[valid_mask]).mean()]
        else:
            errors += [np.abs(gt_depth-ours_depth).mean()]

    if pose_file is None:
        np.savez(os.path.join(gt_dir, "sampled_poses_{}.npz".format(n_imgs)), poses=poses)
    elif not os.path.exists(pose_file):
        np.savez(pose_file, poses=poses)

    if gt_depth_render_file is None:
        np.savez(os.path.join(gt_dir, "gt_depths_{}.npz".format(n_imgs)), depths=gt_depths)
    elif not os.path.exists(gt_depth_render_file):
        np.savez(gt_depth_render_file, depths=gt_depths)

    if depth_render_file is None:
        np.savez(os.path.join(log_dir, "depths_{}_{}.npz".format(suffix, n_imgs)), depths=depths)
    elif not os.path.exists(depth_render_file):
        np.savez(depth_render_file, depths=depths)


    errors = np.array(errors)
    # from m to cm
    print('Depth L1(cm): ', errors.mean()*100)
    return {
        "depth-L1": errors.mean()*100
    }










@EVALUATOR_REGISTRY.register()
class Evaluator3DReconstruction():
    def _parse_cfg(self, cfg):
        self.dataset_name = cfg.DATASET.NAME
        if "Replica" in self.dataset_name:
            self.dataset_type="Replica"
        elif "NeuralRGBD" in self.dataset_name:
            self.dataset_type="RGBD"
        else:
            self.dataset_type=""

    def __init__(self, cfg):
        self._parse_cfg(cfg)


    def eval(self, rec_mesh_filename, gt_mesh_filename, mode,
             use_virt_cams=True, debug_dir=None,recompute_2d_depths=False):

        if mode == '2d':

            assert self.dataset_type in ["Replica", "RGBD"], "Unknown dataset type..."

            if use_virt_cams:
                eval_data_dir = os.path.dirname(gt_mesh_filename)
                unseen_pc_file = os.path.join(eval_data_dir, "gt_pc_unseen.npy")
                pose_file = os.path.join(eval_data_dir, "sampled_poses_1000.npz")
                assert os.path.isfile(unseen_pc_file)
                assert os.path.isfile(pose_file)
                gt_depth_render_file = os.path.join(eval_data_dir, "gt_depths_1000.npz")

                pred_data_dir = os.path.dirname(rec_mesh_filename)
                depth_render_file = os.path.join(pred_data_dir, "depths_virt_cams_1000.npz")
                if recompute_2d_depths and os.path.exists(depth_render_file):
                    os.remove(depth_render_file)

                suffix="virt_cams"
            else:
                eval_data_dir = os.path.dirname(gt_mesh_filename)
                unseen_pc_file = ""
                pose_file = os.path.join(eval_data_dir, "sampled_poses_1000.npz")
                assert os.path.isfile(pose_file)
                gt_depth_render_file = os.path.join(eval_data_dir, "gt_depths_1000.npz")

                pred_data_dir = os.path.dirname(rec_mesh_filename)
                depth_render_file = os.path.join(pred_data_dir, "depths_traj_cams_1000.npz")
                if recompute_2d_depths and os.path.exists(depth_render_file):
                    os.remove(depth_render_file)
                suffix="traj_cams"


            #TODO: this should go in configs.
            if self.dataset_type == "Replica":  # follow NICE-SLAM
                sx, sy, sz, dx, dy, dz = 0.3, 0.7, 0.7, 0.0, 0.0, 0.4
            elif os.path.basename(eval_data_dir) == "complete_kitchen":  # complete_kitchen has special shape
                sx, sy, sz, dx, dy, dz = 0.3, 0.5, 0.5, 1.2, 0.0, 1.8
            else:
                sx, sy, sz, dx, dy, dz = 0.3, 0.6, 0.6, 0.0, 0.0, 0.0

            result = calc_2d_metric_parallel(
                rec_mesh_filename,
                gt_mesh_filename,
                unseen_pc_file, pose_file=pose_file,
                gt_depth_render_file=gt_depth_render_file,
                depth_render_file=depth_render_file,
                n_imgs=1000,
                suffix=suffix,
                not_counting_missing_depth=True,
                sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz
            )

        elif mode == '3d':
            result = calc_3d_metric(rec_mesh_filename, gt_mesh_filename,debug_dir=debug_dir)
        else:
            raise ValueError()

        return result





