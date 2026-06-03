import warnings
warnings.filterwarnings('ignore')

print("⏳  正在初始化 MOABB 并请求 BCI IV-2a 数据集...")

from braindecode.datasets import MOABBDataset

# 自动下载并解析数据
# subject_ids=[3] 代表我们先只拿 9 个人中的 3 号受试者来做初步测试
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[3])

print(f"✅ 数据集加载成功！共提取出 {len(dataset)} 条脑电波数据样本。")

print("\n⏳ 2. 正在构建 EEGNet 脑电波识别神经网络...")
from braindecode.models import EEGNetv4

# 初始化模型
model = EEGNetv4(
    in_chans=22,           # 物理属性：被试者头上贴了 22 个电极（通道）
    n_classes=4,           # 任务属性： 4 分类（左手、右手、脚、舌头）
    input_window_samples=1000, 
    final_conv_length='auto'
)

print(" EEGNet 模型构建成功！")