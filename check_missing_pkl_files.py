#!/usr/bin/env python3

import argparse
import os
import pathlib
from datetime import datetime

def check_missing_pkl_files(output_dir, start_index=6000, end_index=None, verbose=False):
    """
    检查指定目录中从start_index开始的pkl文件是否存在，统计缺失文件数量。
    
    Args:
        output_dir: 输出目录路径
        start_index: 开始检查的索引（默认6000）
        end_index: 结束检查的索引（默认None，表示检查到10000）
        verbose: 是否显示详细信息
    """
    
    output_path = pathlib.Path(output_dir)
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_path}")
        return
    
    print(f"🔍 检查目录: {output_path}")
    print(f"📊 检查范围: {start_index} 到 {end_index if end_index else '无限制'}")
    print("-" * 60)
    
    missing_files = []
    existing_files = []
    total_checked = 0
    
    # 如果没有指定结束索引，先扫描目录找到最大的索引
    if end_index is None:
        max_index = start_index
        for file_path in output_path.glob("*.pkl"):
            try:
                # 提取文件名（去掉.pkl后缀）
                file_name = file_path.stem
                if file_name.isdigit():
                    index = int(file_name)
                    max_index = max(max_index, index)
            except ValueError:
                continue
        
        end_index = max_index + 100  # 检查到最大索引+100，确保覆盖所有可能的文件
        print(f"🔍 自动检测到最大索引: {max_index}, 检查到: {end_index}")
    
    # 检查每个索引对应的pkl文件
    for index in range(start_index, end_index + 1):
        # 使用6位数字格式，如 006001.pkl
        pkl_file = output_path / f"{index:06d}.pkl"
        total_checked += 1
        
        if pkl_file.exists():
            existing_files.append(index)
            if verbose:
                print(f"✅ {index:06d}.pkl - 存在")
        else:
            missing_files.append(index)
            if verbose:
                print(f"❌ {index:06d}.pkl - 缺失")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 检查结果统计:")
    print(f"   总检查文件数: {total_checked}")
    print(f"   存在文件数: {len(existing_files)}")
    print(f"   缺失文件数: {len(missing_files)}")
    print(f"   完整率: {len(existing_files)/total_checked*100:.1f}%")
    
    # 显示缺失文件的详细信息
    if missing_files:
        print(f"\n❌ 缺失的文件列表 (共{len(missing_files)}个):")
        
        # 按连续区间显示缺失文件
        ranges = []
        start = missing_files[0]
        end = missing_files[0]
        
        for i in range(1, len(missing_files)):
            if missing_files[i] == end + 1:
                end = missing_files[i]
            else:
                if start == end:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{end}")
                start = missing_files[i]
                end = missing_files[i]
        
        # 添加最后一个区间
        if start == end:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{end}")
        
        # 显示区间（每行最多5个区间）
        for i in range(0, len(ranges), 5):
            line_ranges = ranges[i:i+5]
            print(f"   {', '.join(line_ranges)}")
        
        # 如果缺失文件太多，只显示前20个和后20个
        if len(missing_files) > 40:
            print(f"\n📋 详细缺失文件列表 (显示前20个和后20个):")
            print(f"   前20个: {missing_files[:20]}")
            print(f"   后20个: {missing_files[-20:]}")
            print(f"   ... (省略中间 {len(missing_files)-40} 个)")
        else:
            print(f"\n📋 详细缺失文件列表:")
            print(f"   {missing_files}")
    else:
        print("\n🎉 所有文件都存在！")
    
    # 显示存在的文件范围
    if existing_files:
        print(f"\n✅ 存在的文件范围: {min(existing_files)} - {max(existing_files)}")
    
    # 保存结果到文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = pathlib.Path(__file__).parent / "logs" / f"missing_files_check_{timestamp}.txt"
    result_file.parent.mkdir(exist_ok=True)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"PKL文件检查结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检查目录: {output_path}\n")
        f.write(f"检查范围: {start_index} 到 {end_index}\n")
        f.write(f"总检查文件数: {total_checked}\n")
        f.write(f"存在文件数: {len(existing_files)}\n")
        f.write(f"缺失文件数: {len(missing_files)}\n")
        f.write(f"完整率: {len(existing_files)/total_checked*100:.1f}%\n\n")
        
        if missing_files:
            f.write(f"缺失文件列表:\n")
            for index in missing_files:
                f.write(f"{index}.pkl\n")
        else:
            f.write("所有文件都存在！\n")
    
    print(f"\n📝 检查结果已保存到: {result_file}")
    
    return len(missing_files), len(existing_files), total_checked

def main():
    parser = argparse.ArgumentParser(description="检查PKL文件是否存在，统计缺失文件数量")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../robot/ik_based/pkl/",
        help="输出目录路径 (默认: ../robot/ik_based/pkl/)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=6000,
        help="开始检查的索引 (默认: 6000)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="结束检查的索引 (默认: 自动检测)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息"
    )
    
    args = parser.parse_args()
    
    print("🔍 PKL文件存在性检查工具")
    print("=" * 60)
    
    missing_count, existing_count, total_count = check_missing_pkl_files(
        args.output_dir, 
        args.start_index, 
        args.end_index, 
        args.verbose
    )
    
    print(f"\n🎯 总结: 缺失 {missing_count} 个文件，存在 {existing_count} 个文件")

if __name__ == "__main__":
    main()
