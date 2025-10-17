#!/usr/bin/env python3

import pathlib

HERE = pathlib.Path(__file__).parent
print(f"Script location: {HERE}")
print(f"Script location absolute: {HERE.absolute()}")

# Calculate base path
BASE_DATA_PATH = HERE.parent.parent / "raw" / "AMASS_SMPLXG"
print(f"Base data path: {BASE_DATA_PATH}")
print(f"Base data path absolute: {BASE_DATA_PATH.absolute()}")
print(f"Base data path exists: {BASE_DATA_PATH.exists()}")

# Check raw directory
raw_dir = HERE.parent.parent / "raw"
print(f"Raw directory: {raw_dir}")
print(f"Raw directory absolute: {raw_dir.absolute()}")
print(f"Raw directory exists: {raw_dir.exists()}")

# Test a specific file path
test_file = BASE_DATA_PATH / "ACCAD" / "Female1General_c3d" / "A10_-_lie_to_crouch_stageii.npz"
print(f"Test file path: {test_file}")
print(f"Test file exists: {test_file.exists()}")




