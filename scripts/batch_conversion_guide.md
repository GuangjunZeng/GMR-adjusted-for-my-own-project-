# 批量PKL到NPZ转换工具使用指南

## 🎯 **功能特点**

基于原始 `29dof_pkl_to_npz.py` 代码，完全保持原有逻辑，新增批量处理功能：

- ✅ **完全保持原代码结构**：所有转换逻辑与原始代码一致
- ✅ **智能跳过已存在文件**：避免重复转换
- ✅ **批量处理**：一次性处理整个目录
- ✅ **详细进度显示**：实时显示处理状态
- ✅ **统计报告**：转换完成后显示详细统计

## 🚀 **使用方法**

### 基本用法
```bash
python 29dof_pkl_to_npz_multiple.py \
  --pkl-dir /path/to/pkl/files \
  --output-dir /path/to/output/npz/files \
  --include-base
```

### 完整参数
```bash
python 29dof_pkl_to_npz_multiple.py \
  --pkl-dir /path/to/pkl/files \
  --output-dir /path/to/output/npz/files \
  --mjcf /path/to/model.xml \
  --include-base \
  --force
```

## 📋 **参数说明**

| 参数 | 必需 | 说明 |
|------|------|------|
| `--pkl-dir` | ✅ | PKL文件所在目录 |
| `--output-dir` | ✅ | NPZ文件输出目录 |
| `--mjcf` | ❌ | MJCF/XML文件路径（推荐） |
| `--include-base` | ❌ | 包含根位置和姿态数据 |
| `--force` | ❌ | 强制重新转换已存在的文件 |

## 🔍 **智能跳过机制**

### 检查逻辑
1. 读取PKL文件名（如：`000000.pkl`）
2. 生成对应NPZ文件名（如：`000000.npz`）
3. 检查输出目录中是否已存在该NPZ文件
4. 如果存在且未使用 `--force`，则跳过转换

### 跳过示例
```
[SKIP] 000000.pkl -> 000000.npz (已存在)
[SKIP] 000001.pkl -> 000001.npz (已存在)
```

## 📊 **输出示例**

### 处理过程
```
🚀 开始批量转换PKL到NPZ...
📁 PKL目录: /path/to/pkl/files
📁 输出目录: /path/to/output/npz/files
🔧 包含基础数据: True
🔄 强制重新转换: False

[INFO] 未提供 --mjcf,将假定 pkl 的列顺序已经与目标顺序一致（不重排）。

📋 找到 5 个PKL文件

[1/5] 处理: 000000.pkl
[PROCESS] 处理: 000000.pkl
[OK] 保存至: /path/to/output/npz/files/000000.npz
     形状: (131, 36)
     列数: 7+29
     精度: float64
     包含数据: ['joints', 'joint_names', 'root_pos', 'root_quat', 'full_data', 'num_frames', 'num_joints', 'include_base']

[2/5] 处理: 000001.pkl
[SKIP] 输出文件已存在: 000001.npz

[3/5] 处理: 000002.pkl
[PROCESS] 处理: 000002.pkl
[OK] 保存至: /path/to/output/npz/files/000002.npz
...
```

### 完成统计
```
============================================================
📊 批量转换完成！
============================================================
✅ 成功转换: 3 个文件
⏭️  跳过文件: 2 个文件
❌ 转换失败: 0 个文件
📁 输出目录: /path/to/output/npz/files

🎉 所有文件处理完成！
```

## 💡 **使用场景**

### 1. 首次批量转换
```bash
python 29dof_pkl_to_npz_multiple.py \
  --pkl-dir ./test_retarget_npz2pkl \
  --output-dir ./npz_output \
  --include-base
```

### 2. 增量转换（跳过已存在）
```bash
python 29dof_pkl_to_npz_multiple.py \
  --pkl-dir ./test_retarget_npz2pkl \
  --output-dir ./npz_output \
  --include-base
```

### 3. 强制重新转换
```bash
python 29dof_pkl_to_npz_multiple.py \
  --pkl-dir ./test_retarget_npz2pkl \
  --output-dir ./npz_output \
  --include-base \
  --force
```

### 4. 使用MJCF文件
```bash
python 29dof_pkl_to_npz_multiple.py \
  --pkl-dir ./test_retarget_npz2pkl \
  --output-dir ./npz_output \
  --mjcf ./model.xml \
  --include-base
```

## ⚠️ **注意事项**

1. **文件命名**：NPZ文件名与PKL文件名相同（仅扩展名不同）
2. **目录权限**：确保对输出目录有写权限
3. **内存使用**：大文件批量处理时注意内存使用
4. **错误处理**：单个文件失败不会影响其他文件的处理

## 🎉 **优势**

- **高效**：智能跳过已存在文件，避免重复工作
- **可靠**：完全基于原始代码，保证转换质量
- **灵活**：支持所有原始参数和功能
- **直观**：详细的进度显示和统计报告

