import os
import sys
sys.path.append("./") # remove when project is compiled
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
from imageio import imwrite
from pathlib import Path

from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser

# -- some tool functions
# -- taken from NGP
def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2,1e-2,3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))



def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)

    root = cfg.META.OUTPUT_DIR
    experiment_name = cfg.META.NAME_EXPERIMENT
    run_id = cfg.META.RUN_ID
    run_id = str(run_id)

    experiment_dir = os.path.join(
        root,
        experiment_name,
        run_id,
    )

    if os.path.isfile(args.input_filename):
        mesh_filename=args.input_filename
    else:
        mesh_filename = os.path.join(
            experiment_dir,
            'mesh_raw.obj'
        )

    mesh = o3d.io.read_triangle_mesh(mesh_filename)

    # -- rescale mesh as if inside NGP
    ngp_scale = cfg.RENDERER.SCALE
    ngp_offset = np.array(cfg.RENDERER.OFFSET)
    #mesh.scale(ngp_scale, center=False)
    mesh.scale(ngp_scale, center=np.zeros(3))
    mesh.translate(ngp_offset, relative=True)

    if not args.output_filename:
        output_filename = os.path.join(
            experiment_dir,
            'mesh_asin_NGP.obj'
        )
        o3d.io.write_triangle_mesh(output_filename, mesh)


    # -- algin mesh with NGP poses -> axis rotation [0,1,2] <- [1,2,0]
    rx=np.pi/2
    R90x = np.array(
        [
            [1,0,0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ]
    )
    Tx = np.eye(4)
    Tx[:3,:3] = R90x
    mesh.transform(Tx)


    rz = np.pi/2
    R90z = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0,0,1],
        ]
    )
    Tz = np.eye(4)
    Tz[:3,:3] = R90z
    mesh.transform(Tz)

    if not args.output_filename:
        output_filename = os.path.join(
            experiment_dir,
            'mesh_rotated_NGP.obj'
        )
        o3d.io.write_triangle_mesh(output_filename, mesh)

    # -- rescale mesh outside of NGP
    mesh.translate(-ngp_offset, relative=True)
    #mesh.scale(1./ngp_scale, center=False)
    mesh.scale(1./ngp_scale, center=np.zeros(3))


    # -- algin mesh with real poses

    # -- UPDATE MESH
    scale = cfg.DATASET.POSES_SCALE
    #mesh.scale(1./scale, center=False)
    mesh.scale(1./scale, center=np.zeros(3))

    # -- UPDATE MESH
    totp = np.array(cfg.DATASET.POSES_CENTER_POINT)
    T = np.eye(4)
    T[:3,3] = totp
    mesh.transform(T)
    #mesh.translate(totp, relative=True) -> this is equivalent


    # -- UPDATE MESH
    up = np.array(cfg.DATASET.POSES_UP_VECTOR)
    #up = up / np.linalg.norm(up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1
    R_inv = np.linalg.inv(R)
    mesh.transform(R_inv)


    # -- UPDATE MESH
    T = np.eye(4)
    T[2,2] = -1.0
    mesh.transform(T)


    # -- UPDATE MESH
    T = np.eye(4)
    T[0,0] = 0.0
    T[0,1] = 1.0
    T[1,1] = 0.0
    T[1,0] = 1.0
    mesh.transform(T)


    # -- final adjustment using ATE rot/trans align.
    ate_filename = os.path.join(
        experiment_dir,
        'eval',
        'results.json'
    )
    if os.path.isfile(ate_filename):
        ate_results = json.load(open(ate_filename, "r"))
        rot = np.array(ate_results["rot"])
        trans = np.array(ate_results["trans"])

        T = np.eye(4)
        T[:3,:3] = rot
        T[:3,3]  = trans[:,0]
        mesh.transform(T)
    else:
        print("/!\ No tracking result filename found -> no adjustment!")


    # -- crop mesh
    mcb = cfg.DATASET.MARCHING_CUBES_BOUND
    mcb = np.array(mcb)

    bb = o3d.geometry.AxisAlignedBoundingBox(mcb[:,0], mcb[:,1])
    mesh = mesh.crop(bb)


    if args.output_filename:
        output_filename=args.output_filename
    else:
        output_filename = os.path.join(
            experiment_dir,
            'mesh_final.obj'
        )

    o3d.io.write_triangle_mesh(output_filename, mesh)
    
    o3d.io.write_triangle_mesh(output_filename[:-4]+".ply", mesh)



if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
