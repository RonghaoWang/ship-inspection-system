import time
import torch
import numpy as np
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 640
N_WARMUP = 50
N_TIMING = 200

# ---------- 已训练模型的 .pt ----------
MODEL_PATHS = {
    "基线 YOLO26x": "/root/autodl-tmp/ultralytics-main/runs/detect/bsl/best.pt",
    "GSConv 实测":  "/root/autodl-tmp/ultralytics-main/runs/detect/11/best.pt",
    "WIoU 实测":    "/root/autodl-tmp/ultralytics-main/runs/detect/v8/best.pt",
}

# ---------- GSConv 结构（从 YAML 构建，不加载权重，仅算参数量） ----------
GSConv_YAML = "/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/26/yolo26-conv.yaml"


def params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def compute_gflops(model):
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return 0.0


def measure_latency(model):
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    model.eval()

    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(dummy)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(N_TIMING):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return times.mean(), times.std(), 1000.0 / times.mean()


print(f"设备: {DEVICE}  |  输入: {INPUT_SIZE}×{INPUT_SIZE}  |  预热 {N_WARMUP} / 计时 {N_TIMING}\n")

# ============================================================
# 先单独算 GSConv YAML 模型的参数量（纯结构，无训练权重）
# ============================================================
print("[GSConv 结构参数量（纯结构定义，scale='x'）]")
try:
    gs_yolo = YOLO("yolo26x-conv.yaml", verbose=False)
    gs_p = params_m(gs_yolo.model)
    gs_g = compute_gflops(gs_yolo.model.to(DEVICE))
    print(f"  GSConv 结构 (x):  {gs_p:.2f}M params  |  {gs_g:.2f} GFLOPs")
except Exception as e:
    print(f"  加载失败: {e}")

# 与原版 YOLO26x 对比
try:
    ref_yolo = YOLO("yolo26x.yaml", verbose=False)
    ref_p = params_m(ref_yolo.model)
    ref_g = compute_gflops(ref_yolo.model.to(DEVICE))
    delta_p = gs_p - ref_p
    delta_g = gs_g - ref_g
    print(f"  原版 YOLO26x:     {ref_p:.2f}M params  |  {ref_g:.2f} GFLOPs")
    print(f"  GSConv 节省:      {abs(delta_p):.2f}M params  "
          f"({abs(delta_p)/ref_p*100:.1f}%)  |  {abs(delta_g):.2f} GFLOPs ({abs(delta_g)/ref_g*100:.1f}%)")
except Exception as e:
    print(f"  对比加载失败: {e}")

print()

# ============================================================
# 测试已训练模型
# ============================================================
print(f"{'模型':<20} {'参数量(M)':<10} {'GFLOPs':<10} {'延迟(ms)':<18} {'FPS':<8}")
print("-" * 70)

for name, path in MODEL_PATHS.items():
    try:
        yolo = YOLO(path, verbose=False)
        yolo.model = yolo.model.to(DEVICE)
        yolo.model.eval()

        p = params_m(yolo.model)
        g = compute_gflops(yolo.model)
        mean_ms, std_ms, fps = measure_latency(yolo.model)

        print(f"{name:<20} {p:<10.2f} {g:<10.2f} {mean_ms:<8.2f} ± {std_ms:<6.2f}  {fps:<8.1f}")

    except Exception as e:
        print(f"{name:<20} 错误: {e}")

print()
print("-" * 70)
print("说明:")
print("  参数量 = 全部可训练参数（卷积核、BN、偏置等）")
print("  GFLOPs = 单张 640×640 推理所需浮点运算量")
print("  延迟   = 200 次推理的均值 ± 标准差（含 GPU 同步）")
print("  FPS    = 1000 / 平均延迟")