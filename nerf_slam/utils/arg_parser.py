import argparse

def default_argument_parser():
    parser = argparse.ArgumentParser()
    # default
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--parent", default="", metavar="FILE", help="path to parent config file")

    # for batch experiments
    parser.add_argument("--output_dir", default="", metavar="DIR", help="path to output expe dir")
    parser.add_argument("--run_id", default="", metavar="ID", help="experiment run id")
    parser.add_argument("--expe_ids", default="", metavar="List", help="list of experiment ids")
    
    # -- 3D mesh recons 
    parser.add_argument("--max_frame_id", default="", metavar="ID", help="max frame ID to consider (for eval)")
    parser.add_argument("--mesh_thresh", default="", metavar="FLOAT", help="threshold for meshing")
    parser.add_argument("--mesh_res", default="", metavar="INT", help="resolution for meshing")

    # -- 3D eval
    parser.add_argument("--eval_rec_mode", default="3d", metavar="STR", help="mode for 3D rec evaluation 3d or 2d")
    parser.add_argument("--recompute_2d_depths", action="store_true", help="re-render 2D depths for eval rec-2d")
     
    # -- Debug
    parser.add_argument("--output_filename", default="", metavar="FILE",help="path to output file (will bypass the default output file path within the code.)")
    parser.add_argument("--input_filename", default="", metavar="FILE",help="path to input file (will bypass the default output file path within the code.)")
    parser.add_argument("--no_output", action="store_true", help="Do not save output (debug)")
    parser.add_argument("--debug", action="store_true", help="toggle debug mode (mostly used for 3D rec eval)")

    return parser


