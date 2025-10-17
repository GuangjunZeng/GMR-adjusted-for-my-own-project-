# NPZ文件验证和可视化工具使用指南

## 🎯 问题解决

由于NPZ文件是二进制格式，无法直接可视化，我们创建了专门的验证工具来确保PKL→NPZ转换的正确性。

## 🛠️ 工具介绍

### 1. `verify_npz_csv.py` - 数据一致性验证工具

**功能：**
- 对比NPZ和CSV文件的数据一致性
- 生成详细的统计报告
- 创建可视化对比图
- 分析NPZ文件结构

**使用方法：**
```bash
# 基本验证
python verify_npz_csv.py --csv 000000.csv --npz 000000.npz

# 带可视化
python verify_npz_csv.py --csv 000000.csv --npz 000000.npz --visualize

# 自定义容差
python verify_npz_csv.py --csv 000000.csv --npz 000000.npz --tolerance 1e-8
```

### 2. `view_npz.py` - NPZ文件查看工具

**功能：**
- 查看NPZ文件内容结构
- 显示数据统计信息
- 可选保存为CSV格式

**使用方法：**
```bash
# 查看NPZ文件内容
python view_npz.py 000000.npz

# 保存为CSV格式
python view_npz.py 000000.npz --save-csv

# 不显示数据预览
python view_npz.py 000000.npz --no-preview
```

## 📊 验证流程

### 步骤1：生成对比数据
```bash
# 生成CSV文件
python 29dof_pkl_to_csv.py --pkl 000000.pkl --out 000000.csv --include-base

# 生成NPZ文件
python 29dof_pkl_to_npz.py --pkl 000000.pkl --out 000000.npz --include-base
```

### 步骤2：验证数据一致性
```bash
# 完整验证（推荐）
python verify_npz_csv.py --csv 000000.csv --npz 000000.npz --visualize
```

### 步骤3：查看验证结果
验证工具会输出：
- ✅ 数据形状对比
- ✅ 数值差异统计
- ✅ 统计信息对比
- ✅ NPZ文件结构分析
- 📊 可视化对比图（如果启用）

## 🔍 输出示例

```
🔍 开始验证NPZ和CSV文件数据一致性...

📂 加载数据...
[CSV] 形状: (131, 36), 数据类型: float64
[NPZ] 包含的键: ['full_data', 'joints', 'root_pos', 'root_quat', 'joint_names', 'num_frames', 'num_joints', 'include_base']
[NPZ] full_data 形状: (131, 36), 数据类型: float64

CSV 统计信息:
  形状: (131, 36)
  数据类型: float64
  最小值: -3.141593
  最大值: 3.141593
  平均值: 0.000000
  标准差: 1.570796

NPZ 统计信息:
  形状: (131, 36)
  数据类型: float64
  最小值: -3.141593
  最大值: 3.141593
  平均值: 0.000000
  标准差: 1.570796

============================================================
数据对比分析
============================================================
CSV形状: (131, 36)
NPZ形状: (131, 36)

数值差异统计:
  最大差异: 0.00e+00
  平均差异: 0.00e+00
  容差阈值: 1.00e-06

✅ 数据完全匹配！

🎉 验证通过！NPZ和CSV数据完全一致！
```

## 📈 可视化功能

验证工具会生成对比图，显示：
- 多个关键列的数据曲线对比
- CSV（蓝色实线）vs NPZ（红色虚线）
- 直观显示数据一致性

## ⚠️ 注意事项

1. **精度要求**：默认容差为1e-6，可根据需要调整
2. **内存使用**：大文件验证时注意内存使用
3. **可视化**：需要matplotlib库支持
4. **文件格式**：确保NPZ文件包含`full_data`键

## 🚀 快速验证命令

```bash
# 一键验证（推荐）
python verify_npz_csv.py --csv 000000.csv --npz 000000.npz --visualize --tolerance 1e-6
```

这样你就可以完全验证NPZ转换的正确性，无需担心数据丢失或格式问题！
