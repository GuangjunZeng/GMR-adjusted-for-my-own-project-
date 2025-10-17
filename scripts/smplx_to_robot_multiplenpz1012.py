#!/usr/bin/env python3

import argparse
import pathlib
import os
import time
import csv
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import torch
import gc

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

from rich import print

HERE = pathlib.Path(__file__).parent


def process_single_npz(smplx_file_path, output_path, robot, SMPLX_FOLDER, verbose=False, visualize=False, record_video=False):
    """
    Process a single NPZ file and save the retargeted motion data.
    Uses the exact same functions as the original scripts.
    """
    print(f"📁 Reading NPZ file: {smplx_file_path}")
    print(f"📁 NPZ file exists: {os.path.exists(smplx_file_path)}")
    
    try:
        # Load SMPLX trajectory - exact same as smplx_to_robot.py
        print(f"✅ NPZ file loaded successfully: {smplx_file_path}")
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            smplx_file_path, SMPLX_FOLDER
        )
        
        # align fps - exact same as smplx_to_robot.py
        tgt_fps = 30
        smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        
        # Initialize the retargeting system - exact same as smplx_to_robot.py
        retarget = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=robot,
        )
        
        # Initialize visualization if needed - exact same as smplx_to_robot.py
        robot_motion_viewer = None
        if visualize:
            robot_motion_viewer = RobotMotionViewer(
                robot_type=robot,
                motion_fps=aligned_fps,
                transparent_robot=0,
                record_video=record_video,
                video_path=f"videos/{robot}_{os.path.basename(smplx_file_path).split('.')[0]}.mp4"
            )
        
        # FPS measurement variables - exact same as smplx_to_robot.py
        fps_counter = 0
        fps_start_time = time.time()
        fps_display_interval = 2.0  # Display FPS every 2 seconds
        
        # Process frames - exact same logic as smplx_to_robot.py
        qpos_list = []
        i = 0
        
        while True:
            # Check if we've processed all frames
            if i >= len(smplx_data_frames):
                break
            
            # FPS measurement - exact same as smplx_to_robot.py
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                if verbose:
                    print(f"Actual rendering FPS: {actual_fps:.2f}")
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
        
        # Close viewer if it was opened
        if robot_motion_viewer is not None:
            robot_motion_viewer.close()
        
        # Convert to numpy arrays - exact same as smplx_to_robot.py
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw - exact same as smplx_to_robot.py
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        
        # Get local body positions using forward kinematics - exact same as smplx_to_robot_dataset.py
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
        
        # Create motion data - exact same structure as both original scripts
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
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        # Clean up GPU memory even on error
        torch.cuda.empty_cache()
        gc.collect()
        return False


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
        help="Number of threads for parallel processing (default: 1, sequential processing).",
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Start reading from this row number (0-based, after header is removed). Default: 0 (start from beginning).",
    )
    
    args = parser.parse_args()
    
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
            if args.start_row > 0:
                print(f"⏭️  Skipped first {args.start_row} rows (start_row={args.start_row})")
                print(f"📊 Processing rows {args.start_row} to {args.start_row + total_data_rows - 1}")
    except Exception as e:
        print(f"Error reading CSV file {args.csv_file}: {e}")
        return
    
    print(f"Found {len(file_pairs)} files to process")
    
    # Process files (sequential or parallel)
    successful = 0
    failed = 0
    
    if args.num_threads == 1:
        # Sequential processing
        for i, (index, npz_path) in enumerate(file_pairs):
            if args.verbose:
                print(f"Processing {i+1}/{len(file_pairs)}: {index}")
            
            # Check if NPZ file exists
            if not os.path.exists(npz_path):
                print(f"NPZ file not found: {npz_path}")
                failed += 1
                continue
            
            # Create output path using index
            output_path = os.path.join(args.save_path, f"{index}.pkl")
            
            # Check if output already exists
            if os.path.exists(output_path):
                if args.verbose:
                    print(f"Output file already exists, skipping: {output_path}")
                successful += 1
                continue
            
            # Process the file
            if process_single_npz(npz_path, output_path, args.robot, SMPLX_FOLDER, args.verbose, args.visualize, args.record_video):
                successful += 1
            else:
                failed += 1
    else:
        # Parallel processing
        print(f"🚀 Using {args.num_threads} threads for parallel processing")
        
        def process_file_wrapper(file_info):
            index, npz_path = file_info
            output_path = os.path.join(args.save_path, f"{index}.pkl")
            
            # Check if NPZ file exists
            if not os.path.exists(npz_path):
                print(f"❌ NPZ file not found: {npz_path}")
                return False, f"NPZ file not found: {npz_path}"
            
            # Check if output already exists
            if os.path.exists(output_path):
                if args.verbose:
                    print(f"⏭️  Output file already exists, skipping: {output_path}")
                return True, f"Already exists: {output_path}"
            
            # Process the file
            if process_single_npz(npz_path, output_path, args.robot, SMPLX_FOLDER, args.verbose, args.visualize, args.record_video):
                return True, f"Success: {output_path}"
            else:
                return False, f"Failed: {output_path}"
        
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_file_wrapper, file_info): file_info for file_info in file_pairs}
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    success, message = future.result()
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
    
    print(f"Processing complete. Successful: {successful}, Failed: {failed}")
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
