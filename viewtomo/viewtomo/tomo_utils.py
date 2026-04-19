"""
tomo_utils.py
-------------
Utility functions for IMOD/AreTomo file parsing, header reading, and subprocess execution.
"""
import os
import sys
import re
import math
import subprocess
import mrcfile
from pathlib import Path

def resolve_template_path(template_name: str) -> Path:
    """
    Search for the template file. 
    1. Check if the path exists as provided (relative to CWD or absolute).
    2. Check if it exists in the package's internal 'templates' directory.
    """
    cmd_path = Path(template_name)
    if cmd_path.exists():
        return cmd_path.resolve()
    
    # Check internal package templates folder
    pkg_template = Path(__file__).parent / "templates" / template_name
    if pkg_template.exists():
        return pkg_template.resolve()
        
    return cmd_path.resolve()

def calculate_thicknesses(mrc_path: str, align_target_nm: float = 150.0, final_target_nm: float = 300.0) -> dict:
    """
    Extracts pixel size using IMOD's header tool and calculates the required thickness
    in unbinned pixels to match the target nanometer thickness.
    """
    try:
        with mrcfile.open(mrc_path, header_only=True) as mrc:
            apix = float(mrc.voxel_size.x)
    except Exception as e:
        raise RuntimeError(f"Failed to read MRC header from {mrc_path}: {e}")

    if apix <= 0:
        raise ValueError(f"Invalid pixel size ({apix} Å) found in {mrc_path}")
    
    def to_even_pixels(target_A, apix_val):
        pixels = int(round(target_A / apix_val))
        return pixels if pixels % 2 == 0 else pixels + 1

    return {
        "apix_angstroms": apix,
        "align_thickness_px": to_even_pixels(align_target_nm * 10.0, apix),
        "final_thickness_px": to_even_pixels(final_target_nm * 10.0, apix)
    }

def run_cmd(cmd: list, cwd=None, log_file=None):
    """
    Executes a system command cleanly.
    If a command fails, raises a RuntimeError instead of exiting to allow batch processing to continue.
    """
    cmd_str = [str(c) for c in cmd]
    try:
        if log_file:
            with open(log_file, "w") as f:
                subprocess.run(cmd_str, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(cmd_str, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        err_msg = f"Command failed in {cwd or '.'} with exit code {e.returncode}: {' '.join(cmd_str)}"
        print(f"\n[ERROR] {err_msg}")
        raise RuntimeError(err_msg) from e

def calculate_imod_binning(n: int, nbin: int):
    """Replicates IMOD's getBinnedSize logic for calculating output boundaries."""
    nbin_out = n // nbin
    irem = n - nbin * nbin_out
    
    if (n % 2) == (nbin_out % 2):
        if irem > 1:
            nbin_out += 2
    else:
        nbin_out += 1
        irem += nbin
        
    ix_offset = 0
    if irem > 1:
        ix_offset = -(nbin - irem // 2)
        
    return nbin_out, ix_offset

def determine_output_size(mrc_path: str, edf_path: str, binning: int) -> str:
    """Calculates the rotation-aware SizeToOutputInXandY for newstack."""
    res = subprocess.run(["header", "-size", str(mrc_path)], capture_output=True, text=True, check=True)
    dims = res.stdout.split()
    nx, ny = int(dims[0]), int(dims[1])
    
    rot = 0.0
    if Path(edf_path).exists():
        with open(edf_path, "r", encoding="utf-8") as f:
            for line in f:
                if "ImageRotationA" in line or "rotation=" in line.lower():
                    parts = line.strip().split("=")
                    if len(parts) > 1:
                        try:
                            rot = float(parts[-1])
                        except ValueError:
                            pass
                            
    if abs(rot) > 45.0:
        nx, ny = ny, nx
        
    nxbin, _ = calculate_imod_binning(nx, binning)
    nybin, _ = calculate_imod_binning(ny, binning)
    
    return f"{nxbin},{nybin}"

def append_or_replace_adoc_keys(adoc_path: Path, replacements: dict):
    """Safely injects dynamic values into an existing .adoc template."""
    with open(adoc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for key, value in replacements.items():
        pattern = re.compile(r"(?m)^\s*" + re.escape(key) + r"\s*=.*$")
        new_line = f"{key}={value}"
        if pattern.search(content):
            content = pattern.sub(new_line, content)
        else:
            if not content.endswith("\n"):
                content += "\n"
            content += new_line + "\n"
            
    with open(adoc_path, 'w', encoding='utf-8') as f:
        f.write(content)