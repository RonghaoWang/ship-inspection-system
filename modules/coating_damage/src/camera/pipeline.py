"""
相机管线初始化、流配置。
"""

from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
    OBAlignMode,
    OBFrameAggregateOutputMode,
    OBPropertyID,
    OBPermissionType,
)

from config import (
    ALIGN_MODE,
    SW_STREAM_WIDTH,
    SW_STREAM_HEIGHT,
    SW_STREAM_FPS,
)


def get_stream_config(pipeline: Pipeline, align_mode: str = ALIGN_MODE):
    """
    根据对齐模式获取深度 / 彩色流的 profile 以及 Config 对象。

    Returns:
        (depth_profile, color_profile, config)  或  None（失败时）
    """
    config = Config()
    try:
        if align_mode == "HW":
            # ── 硬件对齐模式 ──────────────────────────────
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            assert profile_list is not None

            for i in range(len(profile_list)):
                color_profile = profile_list[i]
                if color_profile.get_format() != OBFormat.RGB:
                    continue

                hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(
                    color_profile, OBAlignMode.HW_MODE
                )
                if len(hw_d2c_profile_list) == 0:
                    continue

                hw_d2c_profile = hw_d2c_profile_list[0]
                print("hw_d2c_profile:", hw_d2c_profile)

                config.enable_stream(hw_d2c_profile)
                config.enable_stream(color_profile)
                config.set_align_mode(OBAlignMode.HW_MODE)
                return hw_d2c_profile, color_profile, config

        elif align_mode == "SW":
            # ── 软件对齐模式 ──────────────────────────────
            depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_list.get_video_stream_profile(
                SW_STREAM_WIDTH, SW_STREAM_HEIGHT, OBFormat.Y16, SW_STREAM_FPS
            )
            print("depth profile:", depth_profile)
            config.enable_stream(depth_profile)

            color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_list.get_video_stream_profile(
                SW_STREAM_WIDTH, SW_STREAM_HEIGHT, OBFormat.RGB, SW_STREAM_FPS
            )
            print("color profile:", color_profile)
            config.enable_stream(color_profile)

            config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )
            return depth_profile, color_profile, config

        raise ValueError(f"不支持的 align_mode: {align_mode}，请使用 'HW' 或 'SW'")

    except Exception as e:
        print(e)
        return None


def enable_hw_denoise(pipeline: Pipeline) -> None:
    """
    启用硬件降噪

    Args:
        pipeline: Pipeline对象
    """
    device = pipeline.get_device()
    if device.is_property_supported(
        OBPropertyID.OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
        OBPermissionType.PERMISSION_WRITE
    ):
        # 启用硬件噪声去除滤镜
        device.set_bool_property(
            OBPropertyID.OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL, True
        )
        # 调整硬件噪声去除滤镜的阈值
        device.set_float_property(
            OBPropertyID.OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT, 0.2
        )



