#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from pathlib import Path

# 检查unitree_g1的body links数量
xml_file = Path(__file__).parent.parent / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"

tree = ET.parse(xml_file)
root = tree.getroot()

# 找到所有body标签
bodies = root.findall('.//body')
body_names = [body.get('name') for body in bodies if body.get('name')]

print(f"Total body links found: {len(body_names)}")
print("Body names:")
for i, name in enumerate(body_names, 1):
    print(f"{i:2d}. {name}")

# 检查是否有重复的body names
if len(body_names) != len(set(body_names)):
    print(f"\n⚠️  Warning: Found duplicate body names!")
    from collections import Counter
    duplicates = Counter(body_names)
    for name, count in duplicates.items():
        if count > 1:
            print(f"   '{name}' appears {count} times")
else:
    print(f"\n✅ All body names are unique")