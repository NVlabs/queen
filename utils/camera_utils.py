#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from scene.cameras import Camera, SequentialCamera
from scene.dataset_readers import CameraInfo, SequentialCameraInfo
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, id, cam_info: CameraInfo, resolution_scale: float):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def updateCam(args, image, image_path, frame_idx: int, cam: SequentialCamera, resolution_scale: float):
    orig_w, orig_h = cam.image_width, cam.image_height
    if isinstance(image, torch.Tensor):
        gt_image = image[:3,...]
        loaded_mask = None
        if image.shape[0] == 4:
            loaded_mask = image[3:4,...]
    else:
        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(image, resolution)

        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None

        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]

    cam.update_image(gt_image, image_path, frame_idx, loaded_mask)
    return cam

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def loadSequentialCam(args, id, cam_info:SequentialCameraInfo, resolution_scale, image=None, image_path=None, frame_idx=None):

    # Resize image according to resolution only if already initialized in cam info
    if isinstance(image, torch.Tensor):
        gt_image = image[:3,...]
        loaded_mask = None
        if image.shape[0] == 4:
            loaded_mask = image[3:4,...]
    else:
        if image:
            orig_w, orig_h = image.size
            if args.resolution in [1, 2, 4, 8]:
                resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
            else:  # should be a type that converts to float
                if args.resolution == -1:
                    if orig_w > 1600:
                        global WARNED
                        if not WARNED:
                            print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                            WARNED = True
                        global_down = orig_w / 1600
                    else:
                        global_down = 1
                else:
                    global_down = orig_w / args.resolution

                scale = float(global_down) * float(resolution_scale)
                resolution = (int(orig_w / scale), int(orig_h / scale))

            resized_image_rgb = PILtoTorch(image, resolution)

            gt_image = resized_image_rgb[:3, ...]
            loaded_mask = None

            if resized_image_rgb.shape[1] == 4:
                loaded_mask = resized_image_rgb[3:4, ...]
        else:
            gt_image, loaded_mask = None, None
                 
    return SequentialCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, image_wh=(cam_info.width,cam_info.height),
                  image=gt_image, gt_alpha_mask=loaded_mask, frame_idx=frame_idx,
                  znear = cam_info.znear if cam_info.znear else args.znear, 
                  zfar = cam_info.zfar if cam_info.zfar else args.zfar,
                  image_path=image_path, uid=id, data_device=args.data_device,
                  principle_point=cam_info.principle_point)

def sequentialCameraList_from_camInfos(cam_infos, resolution_scale, args, image_data=None):
    camera_list = []

    for id, c in enumerate(cam_infos):
        if image_data:
            camera_list.append(loadSequentialCam(args, id, c, resolution_scale,
                                                image_data['image'][id], 
                                                image_data['path'][id],
                                                image_data['frame_idx']))
        else:
            camera_list.append(loadSequentialCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera_info : CameraInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera_info.R.transpose()
    Rt[:3, 3] = camera_info.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'width' : camera_info.width,
        'height' : camera_info.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera_info.FovY, camera_info.height),
        'fx' : fov2focal(camera_info.FovX, camera_info.width)
    }
    return camera_entry

