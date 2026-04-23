#!/usr/bin/env python3
"""
viewtomo_align.py
-----------------
Automated pipeline for Cryo-Electron Tomography (Cryo-ET) alignment and reconstruction.
Supports both AreTomo2 (GPU-accelerated) and IMOD/Etomo (CPU patch-tracking) backends.
"""
import os
import sys
import re
import argparse
import shutil
import subprocess
from pathlib import Path

try:
    from viewtomo.mask_ts_outliers import AutoMasker
    from viewtomo import etomo_from_aretomo2 as efa
    from viewtomo.iMOD_comfile import IMOD_comfile
    from viewtomo.tomo_utils import (
        resolve_template_path, calculate_thicknesses, run_cmd, 
        determine_output_size, append_or_replace_adoc_keys
    )
except ImportError:
    try:
        from .mask_ts_outliers import AutoMasker
        from . import etomo_from_aretomo2 as efa
        from .iMOD_comfile import IMOD_comfile
        from .tomo_utils import (
            resolve_template_path, calculate_thicknesses, run_cmd, 
            determine_output_size, append_or_replace_adoc_keys
        )
    except ImportError as e:
        print(f"CRITICAL ERROR: Missing local module. {e}")
        sys.exit(1)

def check_dependencies(engine: str):
    """
    Verifies that all required external command-line tools are available in the system PATH.
    """
    deps = ['header', 'extracttilts', 'newstack', 'etomo', 'makecomfile', 'submfg']
    if engine == 'aretomo2':
        deps.append('AreTomo2')
        
    missing = [cmd for cmd in deps if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError(f"Missing required executables in PATH: {', '.join(missing)}")

class BaseAlignmentEngine:
    """
    Base class for alignment engines.
    Provides shared setup routines, tilt reordering, and outlier masking capabilities.
    """
    def __init__(self, input_mrc: Path, work_dir: Path, params: dict):
        self.orig_mrc = input_mrc.resolve()
        self.work_dir = work_dir
        self.params = params
        self.base_name = self.orig_mrc.stem
        self.linked_mrc = self.work_dir / f"{self.base_name}.mrc"
        self.masked_mrc = self.work_dir / f"{self.base_name}_masked.mrc"

    def setup_workspace(self):
        """Creates the output directory and symlinks the raw MRC to preserve the original."""
        self.work_dir.mkdir(exist_ok=True, parents=True)
        if self.linked_mrc.exists() or self.linked_mrc.is_symlink():
            self.linked_mrc.unlink()
        self.linked_mrc.symlink_to(self.orig_mrc)

    def _check_and_reorder_tilts(self):
        """Parses the MRC header for tilt angles and reorders the stack if unsorted."""
        print(">> Checking tilt angles...")
        res = subprocess.run(["extracttilts", str(self.linked_mrc)], capture_output=True, text=True)
        tilts = []
        for line in res.stdout.splitlines():
            s_line = line.strip()
            if not s_line: continue
            try:
                tilts.append(float(s_line))
            except ValueError:
                continue
        
        if tilts != sorted(tilts):
            print(">> Stack is not sorted. Reordering initial tilt series...")
            tmp_mrc = self.work_dir / f"{self.base_name}_tmp.mrc"
            self.linked_mrc.rename(tmp_mrc)
            run_cmd(["newstack", "-reo", "1", str(tmp_mrc), str(self.linked_mrc)], cwd=self.work_dir)
            tmp_mrc.unlink()
        else:
            print(">> Stack is already sorted.")

    def mask_outliers(self):
        """Executes the outlier masking and inpainting routine."""
        if self.params.get('skip_mask'):
            print(">> Bypassing Outlier Masking (--skip_mask enabled)...")
            if self.masked_mrc.exists() or self.masked_mrc.is_symlink():
                self.masked_mrc.unlink()
            self.masked_mrc.symlink_to(self.linked_mrc)
            return

        print(">> Running Outlier Masking...")
        mask_args = argparse.Namespace(
            input=str(self.linked_mrc),
            output=str(self.masked_mrc),
            binning=6,
            hist_binning=16,
            trial=False,
            debug=self.params.get('debug', False),
            dilation=self.params.get('mask_dilation', 5),
            cut_factor=self.params.get('mask_low_cut', 0.05),
            high_cut_factor=self.params.get('mask_high_cut', 0.05),
            wiggle=self.params.get('wiggle', 1.0),
            dust=self.params.get('dust', 20000),
            workers=self.params.get('workers', 8)
        )
        AutoMasker(mask_args)

class AreTomoEngine(BaseAlignmentEngine):
    """Alignment Engine utilizing AreTomo2 for GPU-accelerated markerless alignment."""
    
    def run(self):
        """Orchestrates the AreTomo2 alignment pipeline."""
        print(f"--- Starting AreTomo2 Pipeline for {self.base_name} ---")
        self.setup_workspace()
        self._check_and_reorder_tilts()
        self.mask_outliers()

        ang_file = self.work_dir / f"{self.base_name}.ang"
        print(">> Extracting tilt angles for AreTomo2...")
        run_cmd(["extracttilts", str(self.linked_mrc), str(ang_file)], cwd=self.work_dir)

        out_mrc = self.work_dir / f"{self.base_name}_aretomo_rec.mrc"
        cmd = [
            "AreTomo2",
            "-InMrc", str(self.masked_mrc),
            "-OutMrc", str(out_mrc),
            "-VolZ", str(self.params['final_thickness_px']),
            "-AlignZ", str(self.params['align_thickness_px']),
            "-OutBin", str(self.params['aretomo_binning']),
            "-AngFile", str(ang_file),
            "-DarkTol", "0.1",
            "-Wbp", "1",
            "-TiltCor", "1",
            "-OutImod", "1"
        ]
        run_cmd(cmd, cwd=self.work_dir)
        self.call_etomo_from_aretomo2()

    def call_etomo_from_aretomo2(self):
        """Invokes the translation module to build IMOD-compatible alignments."""
        print(">> Handing off to etomo_from_aretomo2 for translation and reconstruction...")
        original_argv = sys.argv
        sys.argv = [
            "etomo_from_aretomo2",
            str(self.work_dir),
            "--template", str(self.params['template_path']),
            "--tomo_binning", str(self.params['tomo_binning']),
            "--tomo_thickness", str(self.params['final_thickness_px'])
        ]
        try:
            efa.main()
        except SystemExit as e:
            if e.code != 0 and e.code is not None:
                raise RuntimeError(f"etomo_from_aretomo2 failed with exit code {e.code}")
        finally:
            sys.argv = original_argv
        print(f"--- Success! Pipeline completed for {self.base_name} ---")


class EtomoEngine(BaseAlignmentEngine):
    """Alignment Engine utilizing IMOD/Etomo patch tracking and reconstruction."""
    
    def run(self):
        """Orchestrates the Etomo legacy pipeline."""
        print(f"--- Starting IMOD/Etomo Pipeline for {self.base_name} ---")
        self.setup_workspace()
        self._check_and_reorder_tilts()
        self.mask_outliers()
        self._run_etomo_batch()
        self._run_cryopositioning()
        self._final_reconstruction()
        print(f"--- Success! IMOD pipeline completed. ---")

    def _update_com(self, file_name, param, value):
        """Helper to safely insert or update a parameter in an existing .com file."""
        com_path = self.work_dir / file_name
        if not com_path.exists(): return
        lines = com_path.read_text().splitlines()
        found = False
        for i, line in enumerate(lines):
            if re.match(fr"^{param}\s+", line):
                lines[i] = f"{param} {value}"
                found = True
                break
        if not found:
            lines.append(f"{param} {value}")
        com_path.write_text("\n".join(lines) + "\n")

    def _run_etomo_batch(self):
        """Generates the required .adoc files and runs alignment subprocesses."""
        print(">> Generating IMOD Directives...")
        adoc_path = self.work_dir / f"{self.base_name}.adoc"
        shutil.copy2(self.params['template_path'], adoc_path)
        
        unbinned_thick = self.params['final_thickness_px']
        patch_binning = self.params['aretomo_binning']
        rec_bin = self.params['tomo_binning']
        image_binned = self.params.get('imagebinned', 1)
        scaled_patch_size = int(680 / image_binned)
        patchtrack_border = int((27 * patch_binning) / image_binned)
        
        overrides = {
            "setupset.copyarg.name": self.base_name,
            "setupset.datasetDirectory": str(self.work_dir),
            "setupset.copyarg.stackext": "mrc",
            "setupset.copyarg.pixel": str(self.params['apix_angstroms'] / 10.0),
            "comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY": f"{scaled_patch_size},{scaled_patch_size}",
            "comparam.prenewst.newstack.BinByFactor": str(patch_binning),
            "runtime.Positioning.any.binByFactor": "8",
            "runtime.Positioning.any.thickness": str(unbinned_thick),
            "runtime.AlignedStack.any.binByFactor": str(rec_bin),
            "comparam.tilt.tilt.THICKNESS": str(unbinned_thick),
            "comparam.cryoposition.cryoposition.BinningToApply": "8"
        }
        
        append_or_replace_adoc_keys(adoc_path, overrides)
        run_cmd(["etomo", "--headless", "--directive", adoc_path.name], cwd=self.work_dir)
        run_cmd(["makecomfile", "-root", self.base_name, "-input", "xcorr.com", 
                 "-binning", str(patch_binning), "-change", adoc_path.name, "xcorr_pt.com"], cwd=self.work_dir)
        
        self._update_com("xcorr_pt.com", "BordersInXandY", f"{patchtrack_border},{patchtrack_border}")
        
        for com in ["xcorr.com", "prenewst.com", "xcorr_pt.com", "align.com"]:
            run_cmd(["submfg", com], cwd=self.work_dir)

    def _run_cryopositioning(self):
        """Generates positioning tomograms and calculates pitch offsets to level the lamella."""
        print(">> Running Cryopositioning...")
        adoc_path = self.work_dir / f"{self.base_name}.adoc"
        run_cmd(["makecomfile", "-root", self.base_name, "-thickness", "1200", 
                 "-change", adoc_path.name, "cryoposition.com"], cwd=self.work_dir)
        run_cmd(["submfg", "cryoposition.com"], cwd=self.work_dir)
        run_cmd(["submfg", "tomopitch.com"], cwd=self.work_dir)
        
        pitch_log = (self.work_dir / "tomopitch.log").read_text()
        add_angle = re.search(r"to make level, add\s+([-\d\.]+)", pitch_log).group(1) if "to make level" in pitch_log else "0"
        add_z = re.search(r"added Z shift of\s+([-\d\.]+)", pitch_log).group(1) if "added Z shift" in pitch_log else "0"
        
        def update_additive(param, val):
            align_text = (self.work_dir / "align.com").read_text()
            match = re.search(fr"^{param}\s+([\-\d\.]+)", align_text, re.MULTILINE)
            current = float(match.group(1)) if match else 0.0
            return str(current + float(val))
            
        self._update_com("align.com", "AngleOffset", update_additive("AngleOffset", add_angle))
        self._update_com("align.com", "AxisZshift", update_additive("AxisZshift", add_z))

    def _final_reconstruction(self):
        """Applies level offsets and performs the final back-projection step."""
        print(">> Switching back to unmasked data for final reconstruction...")
        self.linked_mrc.unlink()
        self.linked_mrc.symlink_to(self.orig_mrc)
        self.setup_workspace()
        self._check_and_reorder_tilts()

        edf_path = self.work_dir / f"{self.base_name}.edf"
        final_size = determine_output_size(str(self.linked_mrc), str(edf_path), self.params['tomo_binning'])
        
        newst_lines = (self.work_dir / "newst.com").read_text().splitlines()
        new_lines = []
        for line in newst_lines:
            if line.startswith("$newstack"):
                new_lines.extend([line, f"BinByFactor    {self.params['tomo_binning']}"])
            elif line.startswith("SizeToOutputInXandY"):
                new_lines.append(f"SizeToOutputInXandY    {final_size}")
            else:
                new_lines.append(line)
        (self.work_dir / "newst.com").write_text("\n".join(new_lines))
        run_cmd(["submfg", "newst.com"], cwd=self.work_dir)
        
        self._update_com("tilt.com", "IMAGEBINNED", str(self.params['tomo_binning']))
        self._update_com("tilt.com", "OutputFile", f"{self.base_name}_full_rec.mrc")
        self._update_com("tilt.com", "THICKNESS", str(self.params['final_thickness_px']))
        run_cmd(["submfg", "tilt.com"], cwd=self.work_dir)

def main():
    parser = argparse.ArgumentParser(description="Automated Cryo-ET Alignment and Reconstruction")
    parser.add_argument("inputs", nargs='+', help="Input MRC stack(s)")
    parser.add_argument("--engine", choices=['imod', 'aretomo2'], default='aretomo2', help="Alignment engine, default Aretomo2")
    parser.add_argument("--align_nm", type=float, default=150.0, help="Alignment thickness in nm, default 150")
    parser.add_argument("--final_nm", type=float, default=300.0, help="Reconstruction thickness in nm, default 300")
    parser.add_argument("--aretomo_binning", type=int, default=4, help="Binning for alignment pass, default 4")
    parser.add_argument("--tomo_binning", type=int, default=4, help="Binning for final reconstruction, default 4")
    parser.add_argument("--imagebinned", type=int, default=1, help="Pre-binning factor of the input images (scales patch tracking size), default 1.")
    parser.add_argument("--template", type=str, default="lamella.adoc", help="Path to IMOD system template (.adoc). Uses default 'krios.adoc' in templates")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for python masking, default 8")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and plotting for the masking phase. Writes a useful histogram plot.")
    parser.add_argument("--skip_mask", action="store_true", help="Completely bypass outlier masking phase")
    parser.add_argument("--mask_low_cut", type=float, default=0.05, help="Lower threshold cut factor. Default 0.05")
    parser.add_argument("--mask_high_cut", type=float, default=0.05, help="Upper threshold cut factor. Default 0.05")
    parser.add_argument("--mask_dilation", type=int, default=5, help="Mask boundary dilation in pixels. Default 5")
    parser.add_argument("--wiggle", type=float, default=1.0, help="Masking relaxation factor for high tilts. Default 1.0")
    parser.add_argument("--dust", type=int, default=20000, help="Smallest allowable mask feature size, default 20000.")

    args = parser.parse_args()
    try:
        check_dependencies(args.engine)
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)

    template_path = resolve_template_path(args.template)
    for input_file in args.inputs:
        input_mrc = Path(input_file).resolve()
        if not input_mrc.exists():
            continue
        work_dir = Path.cwd() / input_mrc.stem
        try:
            thick_data = calculate_thicknesses(str(input_mrc), args.align_nm, args.final_nm)
            params = {
                'apix_angstroms': thick_data['apix_angstroms'],
                'align_thickness_px': thick_data['align_thickness_px'],
                'final_thickness_px': thick_data['final_thickness_px'],
                'debug': args.debug,
                'skip_mask': args.skip_mask,
                'mask_low_cut': args.mask_low_cut,
                'mask_high_cut': args.mask_high_cut,
                'mask_dilation': args.mask_dilation,
                'wiggle': args.wiggle,
                'dust': args.dust,
                'aretomo_binning': args.aretomo_binning,
                'tomo_binning': args.tomo_binning,
                'imagebinned': args.imagebinned,
                'template_path': template_path,
                'workers': args.workers
            }
            pipeline = AreTomoEngine(input_mrc, work_dir, params) if args.engine == 'aretomo2' else EtomoEngine(input_mrc, work_dir, params)
            pipeline.run()
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed for {input_mrc.name}: {e}")

if __name__ == "__main__":
    main()