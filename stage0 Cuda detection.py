# 检查 CUDA 是否可用
import os
import torch
import insightface
from whisperx.diarize import DiarizationPipeline
print("torch.cuda.is_available():", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

# insightface 实例（如果已加载）
try:
    arc_model = insightface.app.FaceAnalysis()
    arc_model.prepare(ctx_id=0, det_size=(640, 640))   # ctx_id=0 for GPU; use -1 for CPU
    m = arc_model
    print("insightface ctx_id:", getattr(m, "ctx_id", "<no-ctx_id>"))
except Exception as e:
    print("insightface not loaded:", e)

# pyannote/whisperx pipeline 检查（如果 pipeline 变量名为 `diarization_pipeline` 或类似）
try:
    pipeline = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        device="cuda"
        )
    print(next(pipeline.model.parameters()).device)  # 应该显示 cuda:0
    p = pipeline  # 替换为你实际的变量名
    dev = getattr(p, "cuda", None)
    print("pipeline.device:", dev)
except Exception as e:
    print("no diarization pipeline object available in this scope:", e)

# 运行时查看 GPU 占用
# 在另外一个终端运行: nvidia-smi -l 1