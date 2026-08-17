"""
语义分割模型推理
"""

from ..utils.inference_preprocessing import normalize, letterbox_resize
import cv2
import numpy as np

def predict_ort(ort_session, img, target_size=(512, 512),using_letterbox=False):
    """
    使用 ONNX Runtime 进行推理。
    Args:
        ort_session: 已加载模型的 ONNX Runtime Session 对象
        img: 输入图像 (H×W×C)
        target_size: 目标图像尺寸
        using_letterbox: 是否使用 letterbox_resize 进行预处理，保持宽高比并填充灰条
    Returns:
        pred_mask: 预测的分割掩码 (H×W)，每个像素值为类别 id
    """
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # 预处理：调整大小并归一化
    if using_letterbox:
        # 使用 letterbox_resize，保持原图宽高比，填充灰条
        img_resized, resized_hw, pad_tblr = letterbox_resize(
            img, target_size=target_size
        )
    else:
        # 直接 resize，可能会改变宽高比
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    input_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_img = normalize(input_img)
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)

    # ONNX Runtime 推理
    ort_inputs = {input_name: input_img}
    ort_output = ort_session.run([output_name], ort_inputs)[0]

    if ort_output.ndim == 4:
        # [N, C, H, W] logits 或 [N, 1, H, W] mask
        if ort_output.shape[1] > 1:
            pred_mask = np.argmax(ort_output[0], axis=0).astype(np.uint8)
        pred_mask = ort_output[0, 0].astype(np.uint8)
    elif ort_output.ndim == 3:
        # [N, H, W] 已经是类别 id
        pred_mask = ort_output[0].astype(np.uint8)
    else:
        raise ValueError(f'Unsupported output shape: {ort_output.shape}')

    # 将预测掩码还原回原图大小
    if using_letterbox:
        # pred_mask 目前是 letterbox 大小，且包含灰条[H,W]。需要裁剪掉灰条，并 resize 回原图尺寸。
        # 裁剪 letterbox 灰条区域
        new_h, new_w = resized_hw
        top, bottom, left, right = pad_tblr
        pred_mask = pred_mask[top:top + new_h, left:left + new_w]

    # resize 回原图尺寸
    src_h, src_w = img.shape[:2]
    pred_mask = cv2.resize(pred_mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    return pred_mask


def predict_rknn(rknn_model, img, target_size=(320, 320), using_letterbox=False):
    """
    使用 RKNN 进行推理。
    Args:
        rknn_model: 已加载模型的 RKNN 模型容器对象
        img: 输入图像 (H×W×C)
        target_size: 目标图像尺寸
        using_letterbox: 是否使用 letterbox_resize 进行预处理，保持宽高比并填充灰条
    Returns:
        pred_mask: 预测的分割掩码 (H×W)，每个像素值为类别 id
    """

    # 预处理：调整大小，不用进行归一化，
    # 因为 RKNN 模型通常在转换时已经包含了输入的量化和归一化步骤。
    
    if using_letterbox:
        img_resized, resized_hw, pad_tblr = letterbox_resize(
            img, target_size=target_size
        )

    else:
        # 直接 resize，可能会改变宽高比
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    input_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)

    # RKNN 推理
    rknn_outputs = rknn_model.run([input_img])
    rknn_output = rknn_outputs[0]

    if rknn_output.ndim == 4:
        if rknn_output.shape[1] > 1:
            pred_mask = np.argmax(rknn_output[0], axis=0).astype(np.uint8)
        pred_mask = rknn_output[0, 0].astype(np.uint8)
    elif rknn_output.ndim == 3:
        pred_mask = rknn_output[0].astype(np.uint8)
    else:
        raise ValueError(f'Unsupported output shape: {rknn_output.shape}')
    
    if using_letterbox:
        new_h, new_w = resized_hw
        top, bottom, left, right = pad_tblr
        pred_mask = pred_mask[top:top + new_h, left:left + new_w]

    src_h, src_w = img.shape[:2]
    pred_mask = cv2.resize(pred_mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    return pred_mask


def predict_cv(
    img: np.ndarray,
    canny_low: int = 150,
    canny_high: int = 300,
):
    """
    使用OpenCV方法生成掩码，仅供测试。
    1. Canny 边缘检测
    2. 取反 → 泛洪填充背景为黑色 → 得到目标掩膜
    """
    # 统一为单通道 uint8，提升 Canny 稳定性
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img.copy()
    else:
        raise ValueError("img 必须是 HxW 或 HxWx3 的图像")

    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    # 边缘检测（Canny 输出已是二值 0/255）
    edge_bin = cv2.Canny(gray, canny_low, canny_high)

    # 取反：白色=非边缘可连通区域，黑色=边缘屏障
    non_edges = cv2.bitwise_not(edge_bin)
    
    # 泛洪填充：加 1 像素白边后从外边界填充为黑色，保留内部封闭区域
    h, w = non_edges.shape[:2]
    floodfill = cv2.copyMakeBorder(
        non_edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255
    )
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(floodfill, mask, (0, 0), 0)
    interior_mask = floodfill[1:-1, 1:-1]  # 内部区域 255, 背景 0

    return interior_mask
