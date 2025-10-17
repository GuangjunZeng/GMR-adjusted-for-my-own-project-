#!/usr/bin/env python3

import subprocess
import re
import json
from datetime import datetime

def get_tmux_sessions():
    """获取所有tmux会话"""
    try:
        result = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
        sessions = []
        for line in result.stdout.strip().split('\n'):
            if line:
                session_name = line.split(':')[0]
                sessions.append(session_name)
        return sessions
    except Exception as e:
        print(f"Error getting tmux sessions: {e}")
        return []

def get_gpu_processes():
    """获取GPU进程信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) >= 4:
                    processes.append({
                        'pid': int(parts[0].strip()),
                        'process_name': parts[1].strip(),
                        'gpu_uuid': parts[2].strip(),
                        'used_memory': int(parts[3].strip())
                    })
        return processes
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []

def get_process_info(pid):
    """获取进程详细信息"""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,ppid,cmd', '--no-headers'], capture_output=True, text=True)
        if result.stdout.strip():
            parts = result.stdout.strip().split(None, 2)
            if len(parts) >= 3:
                return {
                    'pid': int(parts[0]),
                    'ppid': int(parts[1]),
                    'cmd': parts[2]
                }
    except Exception as e:
        pass
    return None

def get_tmux_pane_pids(session_name):
    """获取tmux会话中的所有pane PID"""
    try:
        result = subprocess.run(['tmux', 'list-panes', '-t', session_name, '-F', '#{pane_pid}'], capture_output=True, text=True)
        pids = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                pids.append(int(line.strip()))
        return pids
    except Exception as e:
        return []

def find_child_processes(parent_pid):
    """查找父进程的所有子进程，包括孙进程"""
    try:
        # 查找直接子进程
        result = subprocess.run(['ps', '--ppid', str(parent_pid), '-o', 'pid,cmd', '--no-headers'], capture_output=True, text=True)
        children = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split(None, 1)
                if len(parts) >= 2:
                    child_pid = int(parts[0])
                    children.append({
                        'pid': child_pid,
                        'cmd': parts[1]
                    })
                    
                    # 递归查找孙进程
                    grandchildren = find_child_processes(child_pid)
                    children.extend(grandchildren)
        
        return children
    except Exception as e:
        return []

def check_process_status(pid):
    """检查进程是否还在运行"""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,stat', '--no-headers'], capture_output=True, text=True)
        if result.stdout.strip():
            return True
        return False
    except:
        return False

def get_all_python_processes():
    """获取所有Python进程"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        python_processes = []
        for line in result.stdout.strip().split('\n'):
            if 'python' in line and 'smplx_to_robot_multiplenpz.py' in line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = int(parts[1])
                    cmd = ' '.join(parts[10:])
                    python_processes.append({
                        'pid': pid,
                        'cmd': cmd
                    })
        return python_processes
    except Exception as e:
        return []

def main():
    print(f"=== TMUX GPU Usage Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # 获取所有tmux会话
    sessions = get_tmux_sessions()
    print(f"Found {len(sessions)} tmux sessions: {', '.join(sessions)}\n")
    
    # 获取GPU进程信息
    gpu_processes = get_gpu_processes()
    print(f"Found {len(gpu_processes)} GPU processes\n")
    
    # 获取所有Python进程
    all_python_processes = get_all_python_processes()
    print(f"Found {len(all_python_processes)} Python processes\n")
    
    # 分析每个tmux会话
    total_gpu_memory = 0
    unassociated_gpu_memory = 0
    unassociated_processes = []
    
    for session in sessions:
        print(f"=== TMUX Session: {session} ===")
        
        # 获取该会话的所有pane PID
        pane_pids = get_tmux_pane_pids(session)
        print(f"Pane PIDs: {pane_pids}")
        
        session_gpu_memory = 0
        session_processes = []
        
        # 查找每个pane的子进程（递归查找）
        for pane_pid in pane_pids:
            children = find_child_processes(pane_pid)
            for child in children:
                if 'python' in child['cmd'] and 'smplx_to_robot_multiplenpz.py' in child['cmd']:
                    # 检查进程是否还在运行
                    if check_process_status(child['pid']):
                        session_processes.append(child)
                        print(f"  Python process: PID {child['pid']} - {child['cmd'][:100]}...")
                    else:
                        print(f"  Python process: PID {child['pid']} - (已结束)")
        
        # 检查这些进程是否在使用GPU
        associated_pids = set()
        for process in session_processes:
            for gpu_proc in gpu_processes:
                if gpu_proc['pid'] == process['pid']:
                    session_gpu_memory += gpu_proc['used_memory']
                    associated_pids.add(gpu_proc['pid'])
                    print(f"  🎯 GPU Usage: {gpu_proc['used_memory']}MB on GPU {gpu_proc['gpu_uuid'][:8]}...")
        
        total_gpu_memory += session_gpu_memory
        print(f"  📊 Session total GPU memory: {session_gpu_memory}MB")
        print()
    
    # 查找未关联的GPU进程
    print(f"=== UNASSOCIATED GPU PROCESSES ===")
    for gpu_proc in gpu_processes:
        if gpu_proc['pid'] not in [p['pid'] for p in all_python_processes]:
            unassociated_gpu_memory += gpu_proc['used_memory']
            unassociated_processes.append(gpu_proc)
            print(f"PID {gpu_proc['pid']}: {gpu_proc['used_memory']}MB - 未关联到tmux会话")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total tmux sessions: {len(sessions)}")
    print(f"Total GPU memory used by tmux processes: {total_gpu_memory}MB")
    print(f"Unassociated GPU memory: {unassociated_gpu_memory}MB")
    print(f"Total GPU memory: {total_gpu_memory + unassociated_gpu_memory}MB")
    print(f"Average GPU memory per session: {total_gpu_memory/len(sessions) if sessions else 0:.1f}MB")
    
    # 显示所有GPU进程的详细信息
    print(f"\n=== ALL GPU PROCESSES ===")
    for gpu_proc in gpu_processes:
        process_info = get_process_info(gpu_proc['pid'])
        if process_info:
            print(f"PID {gpu_proc['pid']}: {gpu_proc['used_memory']}MB - {process_info['cmd'][:80]}...")
        else:
            print(f"PID {gpu_proc['pid']}: {gpu_proc['used_memory']}MB - Process info not available")
    
    # 添加显存使用汇总
    print(f"\n=== 显存使用汇总 ===")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.stdout.strip():
            total_used = 0
            total_memory = 0
            for line in result.stdout.strip().split('\n'):
                if line and ',' in line:
                    used, total = line.split(',')
                    used = int(used.strip())
                    total = int(total.strip())
                    total_used += used
                    total_memory += total
            
            total_free = total_memory - total_used
            used_percentage = int(total_used * 100 / total_memory) if total_memory > 0 else 0
            free_percentage = int(total_free * 100 / total_memory) if total_memory > 0 else 0
            
            print(f"总显存: {total_memory}MB ({total_memory/1024:.1f}GB)")
            print(f"已使用: {total_used}MB ({total_used/1024:.1f}GB) ({used_percentage}%)")
            print(f"剩余: {total_free}MB ({total_free/1024:.1f}GB) ({free_percentage}%)")
    except Exception as e:
        print(f"无法获取显存汇总信息: {e}")

if __name__ == "__main__":
    main()
