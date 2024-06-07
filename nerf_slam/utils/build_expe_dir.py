import os
import shutil
from pathlib import Path

def build_experiment_directory(cfg):
    root = cfg.META.OUTPUT_DIR
    experiment_name = cfg.META.NAME_EXPERIMENT
    run_id = cfg.META.RUN_ID
    run_id = str(run_id)

    output_dir = os.path.join(
        root,
        experiment_name,
        run_id,
    )
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(output_dir, "configs.yml"), "w") as f:
          f.write(cfg.dump()) 
    
    
    tracker_hyperparams_filename = cfg.TRACKER.TRACKING_HYPERPARAMETERS_FILE
    if os.path.isfile(tracker_hyperparams_filename):
        tracker_hyperparams_filename_dst=os.path.join(output_dir,'tracker_hyperparams.json')
        shutil.copyfile(
            tracker_hyperparams_filename,
            tracker_hyperparams_filename_dst
        )
    
    mapper_hyperparams_filename = cfg.MAPPER.TRACKING_HYPERPARAMETERS_FILE
    if os.path.isfile(mapper_hyperparams_filename):
        mapper_hyperparams_filename_dst=os.path.join(output_dir,'mapper_hyperparams.json')
        shutil.copyfile(
            mapper_hyperparams_filename,
            mapper_hyperparams_filename_dst
        )

    
    ba_hyperparams_filename = cfg.BUNDLE_ADJUSTMENTS.LOCAL_BA.HYPERPARAMETERS_FILE
    if os.path.isfile(ba_hyperparams_filename):
        ba_hyperparams_filename_dst=os.path.join(output_dir,'ba_hyperparams.json')
        shutil.copyfile(
            ba_hyperparams_filename,
            ba_hyperparams_filename_dst
        )
 


    instant_ngp_config_filename = cfg.MODEL.INSTANT_NGP_CONFIG_FILE
    if os.path.isfile(instant_ngp_config_filename):
        instant_ngp_config_filename_dst=os.path.join(output_dir,'instant_ngp_configs.json')
        shutil.copyfile(
            instant_ngp_config_filename,
            instant_ngp_config_filename_dst
        )
