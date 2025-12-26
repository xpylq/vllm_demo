# check_gpu.py - 检查GPU环境
import torch
import subprocess


def check_gpu_environment():
    """检查GPU环境是否满足vLLM要求"""

    print("=" * 60)
    print("GPU环境检查")
    print("=" * 60)

    # 1. 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，vLLM需要NVIDIA GPU")
        return False

    print("✅ CUDA可用")

    # 2. 检查CUDA版本
    cuda_version = torch.version.cuda
    print(f"📌 CUDA版本: {cuda_version}")

    if float(cuda_version.split('.')[0]) < 11:
        print("⚠️  警告: CUDA版本过低，建议11.8+")

    # 3. 检查GPU信息
    gpu_count = torch.cuda.device_count()
    print(f"📌 GPU数量: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / 1024 ** 3

        print(f"\n🎮 GPU {i}: {props.name}")
        print(f"   显存: {total_memory_gb:.1f} GB")
        print(f"   计算能力: {props.major}.{props.minor}")

        # 检查计算能力（建议7.0+，即V100及以上）
        compute_capability = float(f"{props.major}.{props.minor}")
        if compute_capability < 7.0:
            print(f"   ⚠️  计算能力较低，建议使用7.0+的GPU")
        else:
            print(f"   ✅ 计算能力满足要求")

    # 4. 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\n" + "=" * 60)
        print("nvidia-smi 输出:")
        print("=" * 60)
        print(result.stdout)
    except FileNotFoundError:
        print("⚠️  nvidia-smi未找到")

    return True


if __name__ == "__main__":
    check_gpu_environment()