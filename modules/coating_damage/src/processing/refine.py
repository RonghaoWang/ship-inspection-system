import numpy as np
from pyorbbecsdk import OBFormat
import torch

def refine_depth_frame(model, color_frame,depth_frame, K) -> np.ndarray:
    """
    使用模型对深度图进行精细化处理。
    输入原始深度图和对应的彩色图，输出修正后的深度图。
    """

    # 1. 预处理输入数据（如归一化、调整尺寸等）
    # 2. 将数据输入模型进行推理
    # 3. 后处理模型输出（如反归一化、转换数据类型等）

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = color_frame.get_width()
    height = color_frame.get_height()
    color_format = color_frame.get_format()
    if color_format == OBFormat.RGB:
        color_image = np.resize(np.asanyarray(color_frame.get_data()), (height, width, 3))
        image = torch.tensor(color_image / 255, dtype=torch.float32, device=device).permute(
            2, 0, 1
        )[None]

    scale = depth_frame.get_depth_scale()
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        (height, width)
    )
    depth_data = depth_data.astype(np.float32) * scale / 1000.0  # 转换为米
    depth = torch.tensor(depth_data, dtype=torch.float32, device=device)[None]

    intrinsics = K.copy()
    intrinsics[0] /= width  # Normalize fx and cx by width
    intrinsics[1] /= height  # Normalize fy and cy by height
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)[None]

    # Run inference
    output = model.infer(
    image, depth_in=depth, intrinsics=intrinsics, enable_depth_mask=False
    )

    depth_pred = output["depth"]  # Refined depth map
    return depth_pred.squeeze().cpu().numpy() * 1000.0  # Convert back to millimeters
