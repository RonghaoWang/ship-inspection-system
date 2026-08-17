"""
相机内参获取。
"""
import numpy as np

def get_camera_parameters(color_profile, depth_profile) -> np.ndarray:
    """
    从深度流 profile 中提取相机内参矩阵 K (3×3)。
    """
    depth_intrinsics = depth_profile.get_intrinsic()
    print("depth_intrinsics  {}".format(depth_intrinsics))

    depth_distortion = depth_profile.get_distortion()
    print("depth_distortion  {}".format(depth_distortion))

    # Get color internala parameters
    color_intrinsics = color_profile.get_intrinsic()
    print("color_intrinsics  {}".format(color_intrinsics))

    # Get color distortion parameter
    color_distortion = color_profile.get_distortion()
    print("color_distortion  {}".format(color_distortion))    

    # Get external parameters
    extrinsic = depth_profile.get_extrinsic_to(color_profile)
    print("extrinsic  {}".format(extrinsic))


    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy
    cx = depth_intrinsics.cx
    cy = depth_intrinsics.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K
