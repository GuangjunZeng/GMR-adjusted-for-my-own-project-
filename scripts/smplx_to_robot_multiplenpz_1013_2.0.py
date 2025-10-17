#!/usr/bin/env python3

import argparse
import pathlib
import os
import time
import csv
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime

import numpy as np
import torch
import gc

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

from rich import print

HERE = pathlib.Path(__file__).parent


def process_single_npz(smplx_file_path, output_path, robot, SMPLX_FOLDER, verbose=False, visualize=False, record_video=False, log_file=None):
    """
    Process a single NPZ file and save the retargeted motion data.
    Uses the exact same functions as the original scripts.
    """
    # Start timing
    start_time = time.time()
    step_times = {}  # Dictionary to store timing for each step
    avg_time_per_frame = 0  # Initialize for logging
    
    # GPU memory monitoring
    def get_gpu_memory_info():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3     # GB
            return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        return "GPU not available"
    
    print(f"📁 Reading NPZ file: {smplx_file_path}")
    print(f"📁 NPZ file exists: {os.path.exists(smplx_file_path)}")
    if verbose:
        print(f"🔍 {get_gpu_memory_info()}")
    
    try:
        # Step 1: Load SMPLX trajectory - exact same as smplx_to_robot.py
        step_start = time.time()
        print(f"🔄 Step 1/8: Loading SMPLX file...")
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            smplx_file_path, SMPLX_FOLDER
        )
        step_times['load_smplx'] = time.time() - step_start
        print(f"✅ Step 1 completed: {step_times['load_smplx']:.2f}s - NPZ file loaded successfully")
        
        # Step 2: Align FPS - exact same as smplx_to_robot.py
        step_start = time.time()
        print(f"🔄 Step 2/8: Aligning FPS...")
        tgt_fps = 30
        smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        step_times['align_fps'] = time.time() - step_start
        print(f"✅ Step 2 completed: {step_times['align_fps']:.2f}s - FPS aligned to {aligned_fps}")
        
        # Step 3: Initialize the retargeting system - exact same as smplx_to_robot.py
        step_start = time.time()
        print(f"🔄 Step 3/8: Initializing retargeting system...")
        retarget = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=robot,
        )
        step_times['init_retarget'] = time.time() - step_start
        print(f"✅ Step 3 completed: {step_times['init_retarget']:.2f}s - Retargeting system initialized")
        
        # Step 4: Initialize visualization if needed - exact same as smplx_to_robot.py
        step_start = time.time()
        print(f"🔄 Step 4/8: Initializing visualization...")
        robot_motion_viewer = None
        if visualize:
            robot_motion_viewer = RobotMotionViewer(
                robot_type=robot,
                motion_fps=aligned_fps,
                transparent_robot=0,
                record_video=record_video,
                video_path=f"videos/{robot}_{os.path.basename(smplx_file_path).split('.')[0]}.mp4"
            )
        step_times['init_visualization'] = time.time() - step_start
        print(f"✅ Step 4 completed: {step_times['init_visualization']:.2f}s - Visualization initialized")
        
        # FPS measurement variables - exact same as smplx_to_robot.py
        fps_counter = 0
        fps_start_time = time.time()
        fps_display_interval = 2.0  # Display FPS every 2 seconds
        
        # Step 5: Process frames - exact same logic as smplx_to_robot.py
        step_start = time.time()
        print(f"🔄 Step 5/8: Processing motion frames...")
        qpos_list = []
        i = 0
        total_frames = len(smplx_data_frames)
        frame_progress_interval = max(1, total_frames // 20)  # Show progress every 5%
        
        print(f"📊 Total frames to process: {total_frames}")
        
        while True:
            # Check if we've processed all frames
            if i >= len(smplx_data_frames):
                break
            
            # Show progress for every 5% of frames
            if i % frame_progress_interval == 0 or i == total_frames - 1:
                progress_percent = (i / total_frames) * 100
                elapsed_time = time.time() - step_start
                if i > 0:
                    avg_time_per_frame = elapsed_time / i
                    remaining_frames = total_frames - i
                    estimated_remaining_time = avg_time_per_frame * remaining_frames
                    print(f"   📈 Frame {i+1}/{total_frames} ({progress_percent:.1f}%) - Avg: {avg_time_per_frame:.3f}s/frame, ETA: {estimated_remaining_time:.1f}s")
                else:
                    print(f"   📈 Frame {i+1}/{total_frames} ({progress_percent:.1f}%) - Starting...")
            
            # FPS measurement - exact same as smplx_to_robot.py
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                if verbose:
                    print(f"   🔄 Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time
            
            # Update task targets - exact same as smplx_to_robot.py
            smplx_data = smplx_data_frames[i]
            
            # retarget - exact same as smplx_to_robot.py
            qpos = retarget.retarget(smplx_data)
            qpos_list.append(qpos)
            
            # visualize - exact same as smplx_to_robot.py
            if robot_motion_viewer is not None:
                robot_motion_viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False,
                    rate_limit=False,  # For batch processing, we don't use rate limiting
                )
            
            i += 1
        
        step_times['process_frames'] = time.time() - step_start
        avg_time_per_frame = step_times['process_frames'] / len(qpos_list) if len(qpos_list) > 0 else 0
        print(f"✅ Step 5 completed: {step_times['process_frames']:.2f}s - Processed {len(qpos_list)} frames (avg: {avg_time_per_frame:.3f}s/frame)")
        
        # Close viewer if it was opened
        if robot_motion_viewer is not None:
            robot_motion_viewer.close()
        
        # Step 6: Convert to numpy arrays - exact same as smplx_to_robot.py
        step_start = time.time()
        print(f"🔄 Step 6/8: Converting to numpy arrays...")
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw - exact same as smplx_to_robot.py
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        step_times['convert_arrays'] = time.time() - step_start
        print(f"✅ Step 6 completed: {step_times['convert_arrays']:.2f}s - Arrays converted")
        
        # Step 7: Get local body positions using forward kinematics - exact same as smplx_to_robot_dataset.py
        step_start = time.time()
        print(f"🔄 Step 7/8: Computing forward kinematics...")
        device = "cuda:0"
        kinematics_model = KinematicsModel(retarget.xml_file, device=device)
        
        num_frames = root_pos.shape[0]
        fk_root_pos = torch.zeros((num_frames, 3), device=device)
        fk_root_rot = torch.zeros((num_frames, 4), device=device)
        fk_root_rot[:, -1] = 1.0
        
        local_body_pos, _ = kinematics_model.forward_kinematics(
            fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        )
        
        body_names = kinematics_model.body_names
        step_times['forward_kinematics'] = time.time() - step_start
        print(f"✅ Step 7 completed: {step_times['forward_kinematics']:.2f}s - Forward kinematics computed")
        
        # Height adjustment - exact same as smplx_to_robot_dataset.py
        HEIGHT_ADJUST = True
        if HEIGHT_ADJUST:
            # height adjust to ensure the lowerset part is on the ground
            body_pos, _ = kinematics_model.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
                                                            torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
                                                            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)) # TxNx3
            ground_offset = 0.0
            lowerst_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset # make sure motion on the ground
            
        ROOT_ORIGIN_OFFSET = True
        if ROOT_ORIGIN_OFFSET:
            # offset using the first frame
            root_pos[:, :2] -= root_pos[0, :2]
        
        # Step 8: Create motion data and save - exact same structure as both original scripts
        step_start = time.time()
        print(f"🔄 Step 8/8: Creating motion data and saving...")
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos.detach().cpu().numpy(),
            "link_body_list": body_names,
        }
        
        # Save the motion data
        print(f"💾 Saving PKL file: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(motion_data, f)
        
        step_times['save_data'] = time.time() - step_start
        print(f"✅ Step 8 completed: {step_times['save_data']:.2f}s - Motion data saved")
        
        print(f"✅ PKL file saved successfully: {output_path}")
        print(f"📊 PKL file exists: {os.path.exists(output_path)}")
        
        # Print shape information
        print(f"📊 Motion data shapes:")
        print(f"   - FPS: {motion_data['fps']}")
        print(f"   - Root position shape: {motion_data['root_pos'].shape}")
        print(f"   - Root rotation shape: {motion_data['root_rot'].shape}")
        print(f"   - DOF position shape: {motion_data['dof_pos'].shape}")
        print(f"   - Local body position shape: {motion_data['local_body_pos'].shape}")
        print(f"   - Number of frames: {motion_data['root_pos'].shape[0]}")
        print(f"   - Number of body links: {len(motion_data['link_body_list'])}")
        
        if verbose:
            print(f"Successfully processed: {smplx_file_path} -> {output_path}")
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display step-by-step timing summary
        print(f"\n📊 STEP-BY-STEP TIMING SUMMARY:")
        print(f"   Step 1 - Load SMPLX:        {step_times['load_smplx']:.2f}s")
        print(f"   Step 2 - Align FPS:         {step_times['align_fps']:.2f}s")
        print(f"   Step 3 - Init Retarget:     {step_times['init_retarget']:.2f}s")
        print(f"   Step 4 - Init Visualization: {step_times['init_visualization']:.2f}s")
        print(f"   Step 5 - Process Frames:    {step_times['process_frames']:.2f}s")
        print(f"   Step 6 - Convert Arrays:    {step_times['convert_arrays']:.2f}s")
        print(f"   Step 7 - Forward Kinematics: {step_times['forward_kinematics']:.2f}s")
        print(f"   Step 8 - Save Data:         {step_times['save_data']:.2f}s")
        print(f"   ─────────────────────────────────────────")
        print(f"   Total Processing Time:      {processing_time:.2f}s")
        
        # Display processing time
        print(f"⏱️  Processing completed in {processing_time:.2f} seconds")
        
        # Log success details if log_file is provided
        if log_file is not None:
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== SUCCESS LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"Input file: {smplx_file_path}\n")
                    f.write(f"Output file: {output_path}\n")
                    f.write(f"Robot type: {robot}\n")
                    f.write(f"Processing time: {processing_time:.2f} seconds\n")
                    f.write(f"\nSTEP-BY-STEP TIMING:\n")
                    f.write(f"  Step 1 - Load SMPLX:        {step_times['load_smplx']:.2f}s\n")
                    f.write(f"  Step 2 - Align FPS:         {step_times['align_fps']:.2f}s\n")
                    f.write(f"  Step 3 - Init Retarget:     {step_times['init_retarget']:.2f}s\n")
                    f.write(f"  Step 4 - Init Visualization: {step_times['init_visualization']:.2f}s\n")
                    f.write(f"  Step 5 - Process Frames:    {step_times['process_frames']:.2f}s (avg: {avg_time_per_frame:.3f}s/frame)\n")
                    f.write(f"  Step 6 - Convert Arrays:    {step_times['convert_arrays']:.2f}s\n")
                    f.write(f"  Step 7 - Forward Kinematics: {step_times['forward_kinematics']:.2f}s\n")
                    f.write(f"  Step 8 - Save Data:         {step_times['save_data']:.2f}s\n")
                    f.write(f"Status: SUCCESS\n")
                    f.write("=" * 50 + "\n")
            except Exception as log_error:
                print(f"Failed to write success to log file: {log_error}")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        if verbose:
            print(f"🧹 {get_gpu_memory_info()} (after cleanup)")
        
        return True, processing_time, len(qpos_list)
        
    except Exception as e:
        # Calculate processing time even for errors
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"❌ Error processing {smplx_file_path}: {e}")
        print(f"⏱️  Failed after {processing_time:.2f} seconds")
        
        # Display step-by-step timing for failed processing
        if step_times:
            print(f"\n📊 STEP-BY-STEP TIMING (FAILED):")
            for step_name, step_time in step_times.items():
                step_display_name = step_name.replace('_', ' ').title()
                print(f"   {step_display_name}: {step_time:.2f}s")
            print(f"   ─────────────────────────────────────────")
            print(f"   Failed after: {processing_time:.2f}s")
        
        # Log error details if log_file is provided
        if log_file is not None:
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== ERROR LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"Input file: {smplx_file_path}\n")
                    f.write(f"Output file: {output_path}\n")
                    f.write(f"Robot type: {robot}\n")
                    f.write(f"Processing time: {processing_time:.2f} seconds\n")
                    if step_times:
                        f.write(f"\nSTEP-BY-STEP TIMING (FAILED):\n")
                        for step_name, step_time in step_times.items():
                            step_display_name = step_name.replace('_', ' ').title()
                            if step_name == 'process_frames' and avg_time_per_frame > 0:
                                f.write(f"  {step_display_name}: {step_time:.2f}s (avg: {avg_time_per_frame:.3f}s/frame)\n")
                            else:
                                f.write(f"  {step_display_name}: {step_time:.2f}s\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write(f"Error type: {type(e).__name__}\n")
                    f.write("=" * 50 + "\n")
            except Exception as log_error:
                print(f"Failed to write to log file: {log_error}")
        
        # Clean up GPU memory even on error
        torch.cuda.empty_cache()
        gc.collect()
        return False, processing_time, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        help="CSV file containing index and NPZ file paths.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--save_path",
        help="Directory path to save the robot motion files.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="Enable visualization for each processed file.",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record video when visualization is enabled.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads for parallel processing (default: 1, sequential processing). WARNING: High thread counts (>10) may cause GPU memory issues.",
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Start reading from this row number (0-based, after header is removed). Default: 0 (start from beginning).",
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="End reading at this row number (0-based, after header is removed). Default: None (read to end of file).",
    )
    
    args = parser.parse_args()
    
    # GPU memory and thread count warnings
    if args.num_threads > 10:
        print(f"⚠️  WARNING: Using {args.num_threads} threads may cause GPU memory issues!")
        print(f"   - All threads compete for the same GPU (cuda:0)")
        print(f"   - Recommended: Use 1-8 threads for optimal performance")
        print(f"   - High thread counts can cause 10x+ slowdown due to GPU memory competition")
        print(f"   - Consider using fewer threads or CPU-only processing")
        print()
    
    # Provide optimal thread count suggestion
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        suggested_threads = min(8, max(1, int(gpu_memory_gb / 2)))  # Rough estimate: 2GB per thread
        if args.num_threads != suggested_threads:
            print(f"💡 SUGGESTION: For your GPU ({gpu_memory_gb:.1f}GB), consider using {suggested_threads} threads for optimal performance")
            print(f"   Current setting: {args.num_threads} threads")
            print()
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = HERE.parent / f"smplx_to_robot_{timestamp}.log"
    print(f"📝 Log file will be saved to: {log_file_path}")
    
    # Set up paths - exact same as original scripts
    SMPLX_FOLDER = HERE / "assets" / "body_models"
    
    # Set up base path for AMASS_SMPLXG data
    # HERE is the scripts directory (/home/retarget_proj/workbench/data/locomotion/GMR/scripts)
    # We need to go up to GMR, then to locomotion, then to raw
    # CSV paths already start with "AMASS_SMPLXG/", so we only need the raw directory
    BASE_DATA_PATH = HERE.parent.parent / "raw"
    print(f"Script location: {HERE}")
    print(f"Base data path: {BASE_DATA_PATH}")
    print(f"Base data path exists: {BASE_DATA_PATH.exists()}")
    
    # Also check if the raw directory exists
    raw_dir = HERE.parent.parent / "raw"
    print(f"Raw directory: {raw_dir}")
    print(f"Raw directory exists: {raw_dir.exists()}")
    
    # Read CSV file， 所有行都先存储到 file_pairs 列表中
    file_pairs = []
    try:
        with open(args.csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            
            # Check if first row is a header by testing if first column is an integer
            first_row = next(reader, None)
            if first_row and len(first_row) >= 2:
                try:
                    int(first_row[0].strip())
                    # First column is an integer, so this is data, reset to beginning
                    print("First row contains data (integer index), processing from beginning")
                    csvfile.seek(0)
                    reader = csv.reader(csvfile)
                except ValueError:
                    # First column is not an integer, so this is a header, skip it
                    print(f"First row is header (non-integer: '{first_row[0]}'), skipping header row")
                    pass  # Already skipped the header row
            
            row_count = 0
            for row in reader:
                # Skip rows before start_row
                if row_count < args.start_row:
                    row_count += 1
                    continue
                
                # Stop at end_row if specified
                if args.end_row is not None and row_count > args.end_row:
                    break
                
                if len(row) >= 2:  # Ensure we have at least index and file path
                    index = row[0].strip()
                    relative_path = row[-1].strip()  # Last column is the relative file path
                    # Combine base path with relative path from CSV
                    npz_path = BASE_DATA_PATH / relative_path
                    file_pairs.append((index, str(npz_path)))
                
                row_count += 1
            
            # Count total data rows (excluding possible header)
            total_data_rows = len(file_pairs)
            print(f"Total data rows in CSV (excluding header): {total_data_rows}")
            if args.start_row > 0 or args.end_row is not None:
                end_info = f" to {args.end_row}" if args.end_row is not None else " to end"
                print(f"⏭️  Processing rows {args.start_row}{end_info}")
                print(f"📊 Total rows to process: {total_data_rows}")
    except Exception as e:
        print(f"Error reading CSV file {args.csv_file}: {e}")
        return
    
    print(f"Found {len(file_pairs)} files to process")
    
    # Process files (sequential or parallel)
    successful = 0
    failed = 0
    total_processing_time = 0.0
    start_total_time = time.time()
    
    # Efficient tracking for average frames per file
    total_frames_processed = 0
    files_with_frames = 0
    
    if args.num_threads == 1:
        # Sequential processing
        for i, (index, npz_path) in enumerate(file_pairs):
            file_start_time = time.time()
            print(f"\n🔄 Processing file {i+1}/{len(file_pairs)}: {index}")
            print(f"📁 Input file: {npz_path}")
            
            # Check if NPZ file exists
            if not os.path.exists(npz_path):
                print(f"❌ NPZ file not found: {npz_path}")
                failed += 1
                # Log the failure
                try:
                    with open(log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n=== FILE NOT FOUND - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                        f.write(f"File index: {index}\n")
                        f.write(f"NPZ path: {npz_path}\n")
                        f.write(f"Status: FILE NOT FOUND\n")
                        f.write("=" * 50 + "\n")
                except Exception as log_error:
                    print(f"Failed to write to log file: {log_error}")
                continue
            
            # Create output path using index
            output_path = os.path.join(args.save_path, f"{index}.pkl")
            print(f"💾 Output file: {output_path}")
            
            # Check if output already exists
            if os.path.exists(output_path):
                print(f"⏭️  Output file already exists, skipping: {output_path}")
                successful += 1
                # Log the skip
                try:
                    with open(log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n=== SKIPPED (ALREADY EXISTS) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                        f.write(f"File index: {index}\n")
                        f.write(f"Input file: {npz_path}\n")
                        f.write(f"Output file: {output_path}\n")
                        f.write(f"Status: SKIPPED (ALREADY EXISTS)\n")
                        f.write("=" * 50 + "\n")
                except Exception as log_error:
                    print(f"Failed to write to log file: {log_error}")
                continue
            
            # Process the file
            print(f"🚀 Starting processing...")
            success, processing_time, frames_count = process_single_npz(npz_path, output_path, args.robot, SMPLX_FOLDER, args.verbose, args.visualize, args.record_video, str(log_file_path))
            
            # Calculate total file time (including checks)
            file_end_time = time.time()
            total_file_time = file_end_time - file_start_time
            
            total_processing_time += processing_time
            
            # Efficiently track frames for average calculation
            if success and frames_count > 0:
                total_frames_processed += frames_count
                files_with_frames += 1
            
            if success:
                successful += 1
                print(f"✅ File {i+1}/{len(file_pairs)} completed successfully!")
                print(f"⏱️  Processing time: {processing_time:.2f} seconds")
                print(f"⏱️  Total file time: {total_file_time:.2f} seconds")
                print(f"📊 Progress: {successful + failed}/{len(file_pairs)} files processed")
                
                # Log success with timing details
                try:
                    with open(log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n=== FILE SUCCESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                        f.write(f"File index: {index}\n")
                        f.write(f"File number: {i+1}/{len(file_pairs)}\n")
                        f.write(f"Input file: {npz_path}\n")
                        f.write(f"Output file: {output_path}\n")
                        f.write(f"Processing time: {processing_time:.2f} seconds\n")
                        f.write(f"Total file time: {total_file_time:.2f} seconds\n")
                        f.write(f"Status: SUCCESS\n")
                        f.write("=" * 50 + "\n")
                except Exception as log_error:
                    print(f"Failed to write to log file: {log_error}")
            else:
                failed += 1
                print(f"❌ File {i+1}/{len(file_pairs)} failed!")
                print(f"⏱️  Failed after: {processing_time:.2f} seconds")
                print(f"⏱️  Total file time: {total_file_time:.2f} seconds")
                print(f"📊 Progress: {successful + failed}/{len(file_pairs)} files processed")
                
                # Log failure with timing details
                try:
                    with open(log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n=== FILE FAILURE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                        f.write(f"File index: {index}\n")
                        f.write(f"File number: {i+1}/{len(file_pairs)}\n")
                        f.write(f"Input file: {npz_path}\n")
                        f.write(f"Output file: {output_path}\n")
                        f.write(f"Processing time: {processing_time:.2f} seconds\n")
                        f.write(f"Total file time: {total_file_time:.2f} seconds\n")
                        f.write(f"Status: FAILED\n")
                        f.write("=" * 50 + "\n")
                except Exception as log_error:
                    print(f"Failed to write to log file: {log_error}")
            
            # Print progress summary
            remaining_files = len(file_pairs) - (i + 1)
            if remaining_files > 0:
                avg_time_per_file = total_processing_time / (successful + failed) if (successful + failed) > 0 else 0
                estimated_remaining_time = avg_time_per_file * remaining_files
                print(f"📈 Estimated remaining time: {estimated_remaining_time:.2f} seconds ({estimated_remaining_time/60:.1f} minutes)")
                print(f"📈 Average time per file: {avg_time_per_file:.2f} seconds")
            
            print("-" * 80)
    else:
        # Parallel processing
        print(f"🚀 Using {args.num_threads} threads for parallel processing")
        
        def process_file_wrapper(file_info):
            index, npz_path = file_info
            output_path = os.path.join(args.save_path, f"{index}.pkl")
            
            # Check if NPZ file exists
            if not os.path.exists(npz_path):
                print(f"❌ NPZ file not found: {npz_path}")
                return False, f"NPZ file not found: {npz_path}", 0
            
            # Check if output already exists
            if os.path.exists(output_path):
                if args.verbose:
                    print(f"⏭️  Output file already exists, skipping: {output_path}")
                return True, f"Already exists: {output_path}", 0
            
            # Process the file
            success, processing_time, frames_count = process_single_npz(npz_path, output_path, args.robot, SMPLX_FOLDER, args.verbose, args.visualize, args.record_video, str(log_file_path))
            if success:
                return True, f"Success: {output_path} (took {processing_time:.2f}s)", frames_count
            else:
                return False, f"Failed: {output_path} (failed after {processing_time:.2f}s)", 0
        
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_file_wrapper, file_info): file_info for file_info in file_pairs}
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    success, message, frames_count = future.result()
                    # Extract processing time from message for parallel processing
                    if "took" in message:
                        # Extract time from message like "Success: path (took 123.45s)"
                        try:
                            time_str = message.split("took ")[1].split("s)")[0]
                            processing_time = float(time_str)
                            total_processing_time += processing_time
                        except (IndexError, ValueError):
                            pass  # Skip if time extraction fails
                    elif "failed after" in message:
                        # Extract time from message like "Failed: path (failed after 123.45s)"
                        try:
                            time_str = message.split("failed after ")[1].split("s)")[0]
                            processing_time = float(time_str)
                            total_processing_time += processing_time
                        except (IndexError, ValueError):
                            pass  # Skip if time extraction fails
                    
                    # Efficiently track frames for average calculation
                    if success and frames_count > 0:
                        total_frames_processed += frames_count
                        files_with_frames += 1
                    
                    if success:
                        successful += 1
                        if args.verbose:
                            print(f"✅ {message}")
                    else:
                        failed += 1
                        print(f"❌ {message}")
                except Exception as exc:
                    failed += 1
                    print(f"❌ {file_info} generated an exception: {exc}")
    
    # Calculate total execution time
    end_total_time = time.time()
    total_execution_time = end_total_time - start_total_time
    
    # Calculate average frames per file (efficient calculation)
    avg_frames_per_file = total_frames_processed / files_with_frames if files_with_frames > 0 else 0
    
    print(f"Processing complete. Successful: {successful}, Failed: {failed}")
    print(f"⏱️  Total processing time: {total_processing_time:.2f} seconds")
    print(f"⏱️  Total execution time: {total_execution_time:.2f} seconds")
    if successful > 0:
        print(f"⏱️  Average processing time per file: {total_processing_time/successful:.2f} seconds")
    print(f"📊 Average frames per file: {avg_frames_per_file:.1f} frames ({files_with_frames} files with frames)")
    print(f"📊 Total frames processed: {total_frames_processed:,} frames")
    print(f"Saved to {args.save_path}")
    
    # Write final statistics to log
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n=== FINAL STATISTICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Total files processed: {successful + failed}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"单纯加每个文件处理时间累加，但这是并行运算, Total processing time: {total_processing_time:.2f} seconds\n")
            f.write(f"真正的程序运行时间 Total execution time: {total_execution_time:.2f} seconds\n")
            if successful > 0:
                f.write(f"Average processing time per file: {total_processing_time/successful:.2f} seconds\n")
            f.write(f"Average frames per file: {avg_frames_per_file:.1f} frames ({files_with_frames} files with frames)\n")
            f.write(f"Total frames processed: {total_frames_processed:,} frames\n")
            f.write("=" * 50 + "\n")
    except Exception as log_error:
        print(f"Failed to write final statistics to log file: {log_error}")


if __name__ == "__main__":
    main()
