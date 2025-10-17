# SMPL到机器人动作重定向使用说明

## 概述

本项目现在支持将SMPL（Skinned Multi-Person Linear model）人体动作数据重定向到各种机器人模型。这扩展了原有的SMPLX支持，使得可以使用更广泛的SMPL格式动作数据。

## 新增文件

### 1. `scripts/smpl_to_robot.py`
主要的SMPL到机器人动作重定向脚本，仿照`smplx_to_robot.py`的结构。

**使用方法：**
```bash
python scripts/smpl_to_robot.py --smpl_file /path/to/smpl_motion.npz --robot unitree_g1
```

**参数说明：**
- `--smpl_file`: SMPL动作文件路径（.npz格式）
- `--robot`: 目标机器人类型
- `--save_path`: 保存重定向后动作的路径（可选）
- `--loop`: 是否循环播放动作
- `--record_video`: 是否录制视频
- `--rate_limit`: 是否限制播放速率

### 2. 扩展的SMPL处理函数
在`general_motion_retargeting/utils/smpl.py`中新增了以下函数：

- `load_smpl_file_with_model()`: 加载SMPL文件并创建身体模型
- `get_smpl_data()`: 提取单帧关节数据
- `get_smpl_data_offline_fast()`: 离线处理SMPL数据并进行FPS对齐

### 3. 配置文件更新
在`general_motion_retargeting/params.py`中添加了SMPL的IK配置支持，复用了SMPLX的配置（因为关节结构相同）。

## SMPL vs SMPLX 区别

| 特性 | SMPL | SMPLX |
|------|------|-------|
| 关节数量 | 24个主要关节 | 24个主要关节 + 手部关节 |
| 形状参数 | 10个betas | 16个betas（包含手部） |
| 动作参数 | 72个pose参数 | 更多参数（包含手部和面部） |
| 文件格式 | 包含`poses`, `betas`, `trans` | 包含`pose_body`, `root_orient`等 |

## 数据格式要求

SMPL文件应包含以下字段：
- `poses`: 动作参数，形状为(N, 72)，包含根关节朝向和身体关节旋转
- `betas`: 形状参数，形状为(10,)或(N, 10)
- `trans`: 全局平移，形状为(N, 3)
- `gender`: 性别信息（可选）
- `mocap_frame_rate`: 帧率（可选，默认30fps）

## 支持的机器人类型

与SMPLX脚本相同，支持以下机器人：
- unitree_g1, unitree_g1_with_hands
- unitree_h1, unitree_h1_2
- booster_t1, booster_t1_29dof
- stanford_toddy, fourier_n1
- engineai_pm01, kuavo_s45
- hightorque_hi, galaxea_r1pro
- berkeley_humanoid_lite, booster_k1
- pnd_adam_lite, openloong, tienkung

## 使用示例

### 基本使用
```bash
python scripts/smpl_to_robot.py \
    --smpl_file /path/to/motion.npz \
    --robot unitree_g1
```

### 保存重定向结果
```bash
python scripts/smpl_to_robot.py \
    --smpl_file /path/to/motion.npz \
    --robot unitree_g1 \
    --save_path /path/to/output.pkl
```

### 循环播放并录制视频
```bash
python scripts/smpl_to_robot.py \
    --smpl_file /path/to/motion.npz \
    --robot unitree_g1 \
    --loop \
    --record_video
```

## 数据转换

如果你有SMPL格式的数据但需要转换为SMPLX格式，可以使用现有的转换脚本：

```bash
python scripts/smpl_to_smplx.py \
    --input_file /path/to/smpl.npz \
    --output_file /path/to/smplx.npz
```

## 注意事项

1. **身体模型文件**: 确保在`assets/body_models/`目录中有相应的SMPL身体模型文件
2. **数据格式**: 确保SMPL数据文件格式正确，包含必需的字段
3. **关节映射**: SMPL和SMPLX的关节结构基本相同，因此复用了相同的IK配置
4. **性能**: SMPL处理性能与SMPLX相近，因为底层使用相同的处理逻辑

## 故障排除

### 常见问题

1. **ModuleNotFoundError**: 确保安装了所有依赖包
2. **文件格式错误**: 检查SMPL文件是否包含必需的字段
3. **身体模型缺失**: 确保`assets/body_models/`目录中有SMPL模型文件

### 测试脚本

使用提供的测试脚本验证SMPL功能：

```bash
python scripts/test_smpl_functions.py
```

## 技术细节

SMPL处理的关键步骤：
1. 加载SMPL数据并创建身体模型
2. 将SMPL格式转换为内部标准格式
3. 处理betas参数（从10维填充到16维）
4. 进行FPS对齐和插值
5. 使用与SMPLX相同的重定向逻辑

这种设计确保了SMPL和SMPLX数据可以无缝地在同一个系统中处理。
