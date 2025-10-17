#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os

def view_npz_data(npz_path, show_preview=True, save_csv=False):
    """查看NPZ文件内容"""
    print(f"🔍 查看NPZ文件: {npz_path}")
    print("="*60)
    
    # 加载NPZ文件
    data = np.load(npz_path)
    
    print(f"📁 文件包含的键: {list(data.keys())}")
    print()
    
    for key in data.keys():
        value = data[key]
        print(f"🔑 {key}:")
        
        if isinstance(value, np.ndarray):
            print(f"   形状: {value.shape}")
            print(f"   数据类型: {value.dtype}")
            print(f"   数值范围: [{np.min(value):.6f}, {np.max(value):.6f}]")
            
            if show_preview and value.size > 0:
                if value.ndim == 1:
                    print(f"   前5个值: {value[:5]}")
                elif value.ndim == 2:
                    print(f"   前3行3列:\n{value[:3, :3]}")
                elif value.ndim == 3:
                    print(f"   形状预览: {value.shape}")
            
            # 如果是关节名称
            if key == 'joint_names':
                print(f"   关节名称: {list(value)}")
        else:
            print(f"   值: {value}")
        print()
    
    # 保存为CSV（如果请求）
    if save_csv and 'full_data' in data:
        csv_path = npz_path.replace('.npz', '_extracted.csv')
        np.savetxt(csv_path, data['full_data'], fmt='%.6f', delimiter=',')
        print(f"💾 已保存为CSV: {csv_path}")
    
    data.close()

def main():
    ap = argparse.ArgumentParser("查看NPZ文件内容")
    ap.add_argument("npz_path", help="NPZ文件路径")
    ap.add_argument("--no-preview", action="store_true", help="不显示数据预览")
    ap.add_argument("--save-csv", action="store_true", help="保存为CSV文件")
    args = ap.parse_args()
    
    if not os.path.exists(args.npz_path):
        print(f"❌ 文件不存在: {args.npz_path}")
        return
    
    view_npz_data(args.npz_path, 
                  show_preview=not args.no_preview, 
                  save_csv=args.save_csv)

if __name__ == "__main__":
    main()
