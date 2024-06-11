import os
import sys
import numpy as np
import open3d as o3d

sys.path.append("./")
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser



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
            'mesh_final_coslam_culling.ply'
        )

    if not os.path.isfile(mesh_filename):
        raise ValueError

    # - load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    mesh.remove_duplicated_vertices()

    # -- additional cleaning using connected components
    triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 1000
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()

    output_filename = mesh_filename[:-4]+"_connected_comp.ply"

    o3d.io.write_triangle_mesh(output_filename, mesh)
    print("saving file here: ", output_filename)




if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
