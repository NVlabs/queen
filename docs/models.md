Please use the following commands to reproduce the trainings for N3DV and Google-Immersive

---

## N3DV (DyNeRF)

All the runs for N3DV use the same settings, but we include them each for completeness

### coffee_martini
```
python train.py --config configs/dynerf.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/dynerf/coffee_martini -m ./output/trained_n3dv_coffee_martini ; python metrics_video.py -m ./output/trained_n3dv_coffee_martini ;
```

### cook_spinach
```
python train.py --config configs/dynerf.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/dynerf/cook_spinach -m ./output/trained_n3dv_cook_spinach --use_wandb --wandb_run_name trained_n3dv_cook_spinach ; python metrics_video.py -m ./output/trained_n3dv_cook_spinach;
```

### cut_roasted_beef
```
python train.py --config configs/dynerf.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/dynerf/cut_roasted_beef -m ./output/trained_n3dv_cut_roasted_beef --use_wandb --wandb_run_name trained_n3dv_cut_roasted_beef ; python metrics_video.py -m ./output/trained_n3dv_cut_roasted_beef;
```

### sear_steak
```
python train.py --config configs/dynerf.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/dynerf/sear_steak -m ./output/trained_n3dv_sear_steak --use_wandb --wandb_run_name trained_n3dv_sear_steak; python metrics_video.py -m ./output/trained_n3dv_sear_steak;
```

### flame_steak
```
python train.py --config configs/dynerf.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/dynerf/flame_steak -m ./output/trained_n3dv_flame_steak --use_wandb --wandb_run_name trained_n3dv_flame_steak; python metrics_video.py -m ./output/trained_n3dv_flame_steak; 
```

### flame_salmon_1
```
python train.py --config configs/dynerf.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/dynerf/flame_salmon_1 -m ./output/trained_n3dv_flame_salmon_1 --use_wandb --wandb_run_name trained_n3dv_flame_salmon_1; python metrics_video.py -m ./output/trained_n3dv_flame_salmon_1; 
```

## Google Immersive

### cave
```
python train.py --config configs/immersive_cave.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/immersive/cave -m ./output/trained_immersive_cave ; python metrics_video.py -m ./output/trained_immersive_cave ;
```

### exhibit
```
python train.py --config configs/immersive_exhibit.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/immersive/exhibit -m ./output/trained_immersive_exhibit --use_wandb --wandb_run_name trained_immersive_exhibit ; python metrics_video.py -m ./output/trained_immersive_exhibit;
```

### face1
```
python train.py --config configs/immersive_face1.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/immersive/face1 -m ./output/trained_immersive_face1 --use_wandb --wandb_run_name trained_immersive_face1 ; python metrics_video.py -m ./output/trained_immersive_face1;
```

### face2
```
python train.py --config configs/immersive_face2.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/immersive/face2 -m ./output/trained_immersive_face2 --use_wandb --wandb_run_name trained_immersive_face2; python metrics_video.py -m ./output/trained_immersive_face2;
```

### flames
```
python train.py --config configs/immersive_flames.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/immersive/flames -m ./output/trained_immersive_flames --use_wandb --wandb_run_name trained_immersive_flames; python metrics_video.py -m ./output/trained_immersive_flames; 
```

### truck
```
python train.py --config configs/immersive_truck.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/immersive/truck -m ./output/trained_immersive_truck --use_wandb --wandb_run_name trained_immersive_truck; python metrics_video.py -m ./output/trained_immersive_truck; 
```

### welder
```
python train.py --config configs/immersive_welder.yaml --rot_gate_params none --sc_gate_params none --log_images --log_ply --use_xyz_legacy -s data/immersive/welder -m ./output/trained_immersive_welder --use_wandb --wandb_run_name trained_immersive_welder; python metrics_video.py -m ./output/trained_immersive_welder; 
```
