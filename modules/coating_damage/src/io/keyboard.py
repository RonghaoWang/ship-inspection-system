from config import AreaMethod, RuntimeConfig, ESC_KEY
from config import LOAD_MODEL, LOAD_REFINE_MODEL
from src.calc.area_calc import get_timing_stats


def print_keyboard_help() -> None:
    """打印键盘快捷键帮助。"""
    print(
        """
============ Keyboard Help ============
q / ESC : 退出程序
f       : 是否以固定比例显示深度图
r       : 是否精细化深度图（使用模型）
s       : 是否进行区域分割
c       : 循环切换面积算法
a       : 切换是否使用稀疏采样（仅 RANSAC 方法有效）
- / +   : 减小 / 增大 最小区域面积阈值(±100)
e       : 是否保存结果到 Excel 文件
t       : 输出耗时统计并清空
h       : 显示此帮助
=======================================
"""
    )


def print_timing_stats_and_reset() -> None:
    """输出面积计算函数耗时统计并清空累计数据。"""
    stats = get_timing_stats(reset=True)
    if not stats:
        print("[Timing] No timing stats yet.")
        return

    print("[Timing] Area calculation timing stats (ms):")
    for name in sorted(stats.keys()):
        item = stats[name]
        count = int(item.get("count", 0.0))
        total_ms = item.get("total_ms", 0.0)
        avg_ms = item.get("avg_ms", 0.0)
        last_ms = item.get("last_ms", 0.0)
        print(
            f"  - {name}: count={count}, last={last_ms:.3f}, avg={avg_ms:.3f}, total={total_ms:.3f}"
        )
    print("[Timing] Stats reset.")

def next_area_method(method: AreaMethod) -> AreaMethod:
    """循环切换面积计算方法。"""
    methods = [AreaMethod.RANSAC, AreaMethod.DEPTH_CENTER, AreaMethod.AUTO]
    idx = methods.index(method)
    return methods[(idx + 1) % len(methods)]


def handle_key(key: int) -> bool:
    """
    处理键盘事件，返回 True 表示退出。
    """
    cfg = RuntimeConfig
    if key == ord("q") or key == ESC_KEY:
        print("[Key] Quit")
        return True
    elif key == ord("f"):
        cfg.use_fixed_depth_scale = not cfg.use_fixed_depth_scale
        print(f"[Key] Fixed depth scale {'ON' if cfg.use_fixed_depth_scale else 'OFF'}")
    elif key == ord("s"):
        cfg.segment_image = not cfg.segment_image
        print(f"[Key] Segment display {'ON' if cfg.segment_image else 'OFF'}")
    elif key == ord("c"):
        cfg.area_method = next_area_method(cfg.area_method)
        print(f"[Key] Area method = {cfg.area_method.value}")
    elif key == ord("a"):
        cfg.using_sampling = not cfg.using_sampling
        print(f"[Key] Sparse sampling = {cfg.using_sampling}")
    elif key == ord("-"):
        cfg.min_region_area = max(0, cfg.min_region_area - 100)
        print(f"[Key] MIN_REGION_AREA -> {cfg.min_region_area}")
    elif key == ord("+"):
        cfg.min_region_area += 100
        print(f"[Key] MIN_REGION_AREA -> {cfg.min_region_area}")
    elif key == ord("e"):
        cfg.save_in_xlsx = not cfg.save_in_xlsx
        print(f"[Key] Save to Excel = {cfg.save_in_xlsx}")
    elif key == ord("t"):
        print_timing_stats_and_reset()
    elif key == ord("r"):
        if LOAD_MODEL & LOAD_REFINE_MODEL:
            cfg.depth_refinement = not cfg.depth_refinement
            print(f"[Key] Depth Refinement = {cfg.depth_refinement}")
        else:
            print("[Key] Depth Refinement model not loaded, cannot toggle.")

    elif key == ord("h"):
        print_keyboard_help()
    return False
