#!/usr/bin/env python3
"""
批量创建tmux会话的Python脚本
支持检查现有会话、选择性创建、监控功能
"""

import subprocess
import time
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple

class TmuxSessionManager:
    def __init__(self, work_dir: str = "/home/retarget_proj/workbench/data/locomotion/GMR", 
                 num_sessions: int = 6, start_row: int = 9000):
        self.work_dir = work_dir
        self.num_sessions = num_sessions
        self.start_row = start_row
        self.sessions = self._generate_sessions()
    
    def _generate_sessions(self):
        """根据参数生成会话配置，自动找到下一个可用的会话编号"""
        sessions = []
        
        # 获取所有现有的会话名称
        existing_sessions = self.list_sessions()
        existing_numbers = set()
        
        # 提取现有会话的编号
        for session in existing_sessions:
            if session.startswith('smplx_robot'):
                try:
                    # 提取编号部分
                    number_part = session.replace('smplx_robot', '')
                    if number_part.isdigit():
                        existing_numbers.add(int(number_part))
                except:
                    pass
        
        # 找到下一个可用的起始编号
        next_number = 1
        while next_number in existing_numbers:
            next_number += 1
        
        # 生成新的会话配置，确保每个会话都有唯一的编号
        for i in range(self.num_sessions):
            # 确保找到下一个可用的编号
            while next_number in existing_numbers:
                next_number += 1
            
            session_name = f"smplx_robot{next_number:03d}"
            
            #! 每个会话的起始行递增10
            current_start_row = self.start_row + (i * 50)
            
            # 根据会话数量调整线程数
            if self.num_sessions <= 3:
                num_threads = 1
            elif self.num_sessions <= 6:
                num_threads = 1
            elif self.num_sessions <= 10:
                num_threads = 1
            else:
                num_threads = 1
            
            sessions.append({
                "name": session_name,
                "start_row": current_start_row,
                "num_threads": num_threads,
                "description": f"从第{current_start_row}行开始处理"
            })
            
            # 将当前编号标记为已使用，继续下一个
            existing_numbers.add(next_number)
            next_number += 1
        
        return sessions
    
    def run_command(self, cmd: str, check: bool = True) -> Tuple[bool, str]:
        """运行命令并返回结果"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.work_dir)
            if check and result.returncode != 0:
                return False, result.stderr
            return True, result.stdout
        except Exception as e:
            return False, str(e)
    
    def session_exists(self, session_name: str) -> bool:
        """检查tmux会话是否存在"""
        success, output = self.run_command(f"tmux has-session -t {session_name}", check=False)
        return success and output.strip() == ""
    
    def list_sessions(self) -> List[str]:
        """列出所有tmux会话"""
        success, output = self.run_command("tmux list-sessions", check=False)
        if success:
            sessions = []
            for line in output.strip().split('\n'):
                if line.strip():
                    session_name = line.split(':')[0]
                    sessions.append(session_name)
            return sessions
        return []
    
    def get_gpu_status(self) -> Dict:
        """获取GPU状态"""
        success, output = self.run_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", check=False)
        if success and output.strip():
            lines = output.strip().split('\n')
            gpu_info = []
            for line in lines:
                if ',' in line:
                    used, total = line.split(',')
                    used = int(used.strip())
                    total = int(total.strip())
                    percentage = int(used * 100 / total) if total > 0 else 0
                    gpu_info.append({
                        'used_mb': used,
                        'total_mb': total,
                        'percentage': percentage
                    })
            return {'gpus': gpu_info}
        return {'gpus': []}
    
    def create_session(self, session_config: Dict) -> Tuple[bool, str]:
        """创建单个tmux会话"""
        session_name = session_config['name']
        start_row = session_config['start_row']
        num_threads = session_config['num_threads']
        description = session_config['description']
        
        # 构建命令
        cmd = f"""tmux new-session -d -s {session_name} "python scripts/smplx_to_robot_multiplenpz.py --csv_file ../raw/manifest.csv --save_path ../robot/ik_based/pkl/ --start_row {start_row} --num_threads {num_threads}" """
        
        print(f"创建会话: {session_name}")
        print(f"  - 起始行: {start_row}")
        print(f"  - 线程数: {num_threads}")
        print(f"  - 描述: {description}")
        
        success, output = self.run_command(cmd)
        if success:
            print(f"✓ 会话 {session_name} 创建成功")
            return True, "创建成功"
        else:
            print(f"✗ 会话 {session_name} 创建失败: {output}")
            return False, output
    
    def show_config(self):
        """显示当前配置信息"""
        print("=== 会话配置信息 ===")
        print(f"会话数量: {self.num_sessions}")
        print(f"起始行: {self.start_row}")
        print(f"工作目录: {self.work_dir}")
        
        # 显示现有会话
        existing_sessions = self.list_sessions()
        if existing_sessions:
            print(f"\n现有会话: {', '.join(existing_sessions)}")
        else:
            print("\n现有会话: 无")
        
        print("\n将要创建的会话:")
        for session in self.sessions:
            print(f"  - {session['name']}: 起始行 {session['start_row']}, 线程数 {session['num_threads']}, {session['description']}")
        print()

    def create_all_sessions(self) -> Dict:
        """创建所有会话"""
        print("开始批量创建tmux会话...")
        self.show_config()
        results = {'created': 0, 'skipped': 0, 'failed': 0, 'details': []}
        
        for session_config in self.sessions:
            session_name = session_config['name']
            
            # 直接创建会话（配置已经确保了唯一性）
            success, message = self.create_session(session_config)
            if success:
                results['created'] += 1
                results['details'].append({'session': session_name, 'status': 'created', 'message': message})
            else:
                results['failed'] += 1
                results['details'].append({'session': session_name, 'status': 'failed', 'message': message})
            
            # 添加延迟避免冲突
            time.sleep(1)
        
        # 显示显存使用汇总
        self._show_memory_summary()
        
        return results
    
    def _show_memory_summary(self):
        """显示显存使用汇总"""
        print("\n=== 显存使用汇总 ===")
        
        # 获取所有会话信息
        all_sessions = self.list_sessions()
        print(f"总会话数量: {len(all_sessions)}")
        if all_sessions:
            print(f"所有会话: {', '.join(all_sessions)}")
        
        # 获取GPU状态
        gpu_status = self.get_gpu_status()
        if gpu_status['gpus']:
            # 计算所有GPU的总使用量和总容量
            total_used = sum(gpu['used_mb'] for gpu in gpu_status['gpus'])
            total_memory = sum(gpu['total_mb'] for gpu in gpu_status['gpus'])
            total_free = total_memory - total_used
            
            # 但显示时以单个GPU为基准，因为每个程序只能使用一个GPU
            single_gpu_memory = gpu_status['gpus'][0]['total_mb']  # 单个GPU显存
            single_gpu_used = gpu_status['gpus'][0]['used_mb']      # 单个GPU使用量
            
            print(f"总显存: {total_memory}MB ({total_memory/1024:.1f}GB) (所有{len(gpu_status['gpus'])}个GPU)")
            print(f"已使用: {total_used}MB ({total_used/1024:.1f}GB) (所有GPU总和)")
            print(f"剩余: {total_free}MB ({total_free/1024:.1f}GB) (所有GPU总和)")
            
            # 显示每个GPU的详细信息（以单个GPU为基准）
            print(f"\n各GPU使用情况:")
            for i, gpu in enumerate(gpu_status['gpus']):
                print(f"  GPU {i}: {gpu['used_mb']}MB ({gpu['used_mb']/1024:.1f}GB) / {gpu['total_mb']}MB ({gpu['total_mb']/1024:.1f}GB) ({gpu['percentage']}%)")
            
            # 显示单个GPU的使用情况（更符合实际使用场景）
            print(f"\n单个GPU使用情况 (以GPU 0为例):")
            print(f"  单个GPU显存: {single_gpu_memory}MB ({single_gpu_memory/1024:.1f}GB)")
            print(f"  单个GPU使用: {single_gpu_used}MB ({single_gpu_used/1024:.1f}GB) ({gpu_status['gpus'][0]['percentage']}%)")
            print(f"  单个GPU剩余: {single_gpu_memory - single_gpu_used}MB ({(single_gpu_memory - single_gpu_used)/1024:.1f}GB)")
        else:
            print("无法获取GPU信息")
    
    def create_missing_sessions(self) -> Dict:
        """只创建不存在的会话"""
        print("检查并创建缺失的会话...")
        results = {'created': 0, 'skipped': 0, 'failed': 0, 'details': []}
        
        for session_config in self.sessions:
            session_name = session_config['name']
            
            if self.session_exists(session_name):
                print(f"⚠ 会话 {session_name} 已存在，跳过")
                results['skipped'] += 1
                results['details'].append({'session': session_name, 'status': 'skipped', 'message': '已存在'})
                continue
            
            success, message = self.create_session(session_config)
            if success:
                results['created'] += 1
                results['details'].append({'session': session_name, 'status': 'created', 'message': message})
            else:
                results['failed'] += 1
                results['details'].append({'session': session_name, 'status': 'failed', 'message': message})
            
            time.sleep(1)
        
        # 显示显存使用汇总
        self._show_memory_summary()
        
        return results
    
    def show_status(self):
        """显示状态信息"""
        print("=== TMUX会话状态 ===")
        sessions = self.list_sessions()
        if sessions:
            for session in sessions:
                print(f"  - {session}")
        else:
            print("  没有活动的tmux会话")
        
        print("\n=== GPU使用情况 ===")
        gpu_status = self.get_gpu_status()
        for i, gpu in enumerate(gpu_status['gpus']):
            print(f"  GPU {i}: {gpu['used_mb']}MB ({gpu['used_mb']/1024:.1f}GB) / {gpu['total_mb']}MB ({gpu['total_mb']/1024:.1f}GB) ({gpu['percentage']}%)")
        
        # 添加显存使用汇总
        if gpu_status['gpus']:
            print("\n=== 显存使用汇总 ===")
            total_used = sum(gpu['used_mb'] for gpu in gpu_status['gpus'])
            total_memory = sum(gpu['total_mb'] for gpu in gpu_status['gpus'])
            total_free = total_memory - total_used
            total_percentage = int(total_used * 100 / total_memory) if total_memory > 0 else 0
            free_percentage = int(total_free * 100 / total_memory) if total_memory > 0 else 0
            
            print(f"总显存: {total_memory}MB ({total_memory/1024:.1f}GB)")
            print(f"已使用: {total_used}MB ({total_used/1024:.1f}GB) ({total_percentage}%)")
            print(f"剩余: {total_free}MB ({total_free/1024:.1f}GB) ({free_percentage}%)")
    
    def monitor_sessions(self, interval: int = 5):
        """监控模式"""
        print(f"进入监控模式 (每{interval}秒刷新，按Ctrl+C退出)")
        try:
            while True:
                print(f"\n=== TMUX会话监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                self.show_status()
                print(f"\n按Ctrl+C退出监控...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n监控已停止")

def main():
    parser = argparse.ArgumentParser(description='批量创建tmux会话')
    parser.add_argument('-l', '--list', action='store_true', help='列出所有会话状态')
    parser.add_argument('-c', '--create', action='store_true', help='创建所有会话')
    parser.add_argument('-s', '--status', action='store_true', help='显示会话状态和GPU使用情况')
    parser.add_argument('-m', '--monitor', action='store_true', help='监控模式')
    parser.add_argument('--create-missing', action='store_true', help='只创建不存在的会话')
    parser.add_argument('--interval', type=int, default=5, help='监控模式刷新间隔(秒)')
    parser.add_argument('--work-dir', default='/home/retarget_proj/workbench/data/locomotion/GMR', help='工作目录')
    parser.add_argument('--num-sessions', type=int, default=6, help='要创建的会话数量 (默认: 6)')
    parser.add_argument('--start-row', type=int, default=9000, help='起始行号 (默认: 9000)')
    parser.add_argument('--show-config', action='store_true', help='显示配置信息')
    
    args = parser.parse_args()
    
    manager = TmuxSessionManager(args.work_dir, args.num_sessions, args.start_row)
    
    if args.show_config:
        manager.show_config()
    
    elif args.list:
        sessions = manager.list_sessions()
        if sessions:
            print("活动的tmux会话:")
            for session in sessions:
                print(f"  - {session}")
        else:
            print("没有活动的tmux会话")
    
    elif args.create:
        results = manager.create_all_sessions()
        print(f"\n=== 创建结果 ===")
        print(f"成功创建: {results['created']}")
        print(f"跳过(已存在): {results['skipped']}")
        print(f"创建失败: {results['failed']}")
    
    elif args.create_missing:
        results = manager.create_missing_sessions()
        print(f"\n=== 创建结果 ===")
        print(f"新创建: {results['created']}")
        print(f"跳过(已存在): {results['skipped']}")
        print(f"创建失败: {results['failed']}")
    
    elif args.status:
        manager.show_status()
    
    elif args.monitor:
        manager.monitor_sessions(args.interval)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
