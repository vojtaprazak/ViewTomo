#!/usr/bin/env python3
"""
viewtomo_align.py
-----------------
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
    Raises a RuntimeError if any critical dependency is missing.
    """
    deps = ['header', 'extracttilts', 'newstack', 'etomo', 'makecomfile', 'submfg']
    if engine == 'aretomo2':
        deps.append('AreTomo2')
        
    missing = [cmd for cmd in deps if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError(f"Missing required executables in PATH: {', '.join(missing)}\n"
                           f"Please ensure IMOD and AreTomo2 are loaded in your environment.")

class BaseAlignmentEngine:
    """
    Base class for alignment engines.
    Provides shared setup routines and outlier masking capabilities.
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

    def mask_outliers(self):
        """
        Executes the outlier masking routine.
        If the 'skip_mask' parameter is True, bypasses the masking process entirely
        and symlinks the original MRC file to the masked target.
        """
        if self.params.get('skip_mask'):
            print(">> Bypassing Outlier Masking (--skip_mask enabled)...")
            if self.masked_mrc.exists() or self.masked_mrc.is_symlink():
                self.masked_mrc.unlink()
            self.masked_mrc.symlink_to(self.orig_mrc)
            # Update the linked input to point to the bypassed "masked" file
            self.linked_mrc.unlink()
            self.linked_mrc.symlink_to(self.masked_mrc)
            return

        print(">> Running Outlier Masking...")
        mask_args = argparse.Namespace(
            input=str(self.linked_mrc),
            output=str(self.masked_mrc),
            binning=6,
            hist_binning=16,
            trial=False,
            debug=self.params.get('debug', False),
            dilation=self.params.get('mask_dilation', 6),
            softness=30.0,
            cut_factor=self.params.get('mask_low_cut', 0.25),
            high_cut_factor=self.params.get('mask_high_cut', 0.25),
            wiggle=self.params.get('wiggle', 1.0),
            pretilt=None,
            threshold=0.15,
            dust=20000,
            workers=self.params.get('workers', 8)
        )
        AutoMasker(mask_args)
        self.linked_mrc.unlink()
        self.linked_mrc.symlink_to(self.masked_mrc)

class AreTomoEngine(BaseAlignmentEngine):
    """Alignment Engine utilizing AreTomo2 for GPU-accelerated robust markerless alignment."""
    
    def run(self):
        """Orchestrates the AreTomo2 alignment pipeline and translation logic."""
        print(f"--- Starting AreTomo2 Pipeline for {self.base_name} ---")
        self.setup_workspace()
        self.mask_outliers()

        ang_file = self.work_dir / f"{self.base_name}.ang"
        print(">> Extracting tilt angles from raw stack...")
        run_cmd(["extracttilts", str(self.orig_mrc), str(ang_file)], cwd=self.work_dir)

        out_mrc = self.work_dir / f"{self.base_name}_aretomo_rec.mrc"
        print(f">> Running AreTomo2 Alignment (Binning: {self.params['aretomo_binning']})...")
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
        
        log_path = self.work_dir / "aretomo2.log"
        run_cmd(cmd, cwd=self.work_dir, log_file=log_path)
        
        self.linked_mrc.unlink()
        self.linked_mrc.symlink_to(self.orig_mrc)
        
        self.call_etomo_from_aretomo2()

    def call_etomo_from_aretomo2(self):
        """Mocks CLI arguments and invokes the translation module to build IMOD alignments."""
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

    def _check_and_reorder_tilts(self):
        """Parses the MRC header for tilt angles and reorders the stack by angle if unsorted."""
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
            print(">> Stack is not sorted. Reordering...")
            tmp_mrc = self.work_dir / f"{self.base_name}_tmp.mrc"
            self.linked_mrc.rename(tmp_mrc)
            run_cmd(["newstack", "-reo", "1", tmp_mrc, self.linked_mrc])
            tmp_mrc.unlink()

    def _run_etomo_batch(self):
        """Generates the required .adoc files and runs alignment subprocesses."""
        print(">> Generating IMOD Directives...")
        adoc_path = self.work_dir / f"{self.base_name}.adoc"
        shutil.copy2(self.params['template_path'], adoc_path)
        
        patch_binning = self.params['aretomo_binning']
        pos_thick = self.params['final_thickness_px'] 
        positioning_binning = 8
        positioning_thickness = 1200
        defocus = 50000
        
        # Scale the patch tracking sizes and borders by the pre-binning factor of the input images
        image_binned = self.params.get('imagebinned', 1)
        base_patch_size = 680
        scaled_patch_size = int(base_patch_size / image_binned)
        patchtrack_border = int((27 * patch_binning) / image_binned)

        overrides = {
            "setupset.copyarg.name": self.base_name,
            "setupset.datasetDirectory": str(self.work_dir),
            "setupset.copyarg.stackext": "mrc",
            "setupset.scanHeader": "1",
            "setupset.copyarg.pixel": str(self.params['apix_angstroms'] / 10.0),
            "setupset.copyarg.gold": "50",
            "setupset.copyarg.dual": "0",
            "comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY": f"{scaled_patch_size},{scaled_patch_size}",
            "comparam.prenewst.newstack.BinByFactor": str(patch_binning),
            "runtime.Positioning.any.binByFactor": str(positioning_binning),
            "runtime.Positioning.any.thickness": str(positioning_thickness),
            "setupset.copyarg.defocus": str(defocus),
            "runtime.AlignedStack.any.binByFactor": str(patch_binning),
            "comparam.tilt.tilt.THICKNESS": str(pos_thick),
            "comparam.cryoposition.cryoposition.BinningToApply": str(positioning_binning)
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
        
        add_angle, add_z, x_tilt, new_thick = "0", "0", "0", str(self.params['final_thickness_px'])
        
        for line in pitch_log.splitlines():
            if "to make level, add" in line:
                m = re.search(r"to make level, add\s+([-\d\.]+)", line)
                if m: add_angle = m.group(1)
            elif "imply added Z shift" in line and "x-tilted" in line:
                m1 = re.search(r"added Z shift of\s+([-\d\.]+)", line)
                if m1: add_z = m1.group(1)
                m2 = re.search(r"set to\s+([-\d\.]+)", line)
                if m2: new_thick = m2.group(1)
            elif "added X-axis tilt of" in line:
                m = re.search(r"added X-axis tilt of\s+([-\d\.]+)", line)
                if m: x_tilt = m.group(1)
                
        def update_additive(param, extracted_val):
            align_text = (self.work_dir / "align.com").read_text()
            match = re.search(fr"^{param}\s+([\-\d\.]+)", align_text, re.MULTILINE)
            current = float(match.group(1)) if match else 0.0
            return str(current + float(extracted_val))
            
        self._update_com("align.com", "AngleOffset", update_additive("AngleOffset", add_angle))
        self._update_com("align.com", "AxisZshift", update_additive("AxisZshift", add_z))

    def _final_reconstruction(self):
        """Applies level offsets and performs the final back-projection step."""
        print(">> Switching back to unmasked data for final reconstruction...")
        self.linked_mrc.unlink()
        self.linked_mrc.symlink_to(self.orig_mrc)
        
        edf_path = self.work_dir / f"{self.base_name}.edf"
        final_size = determine_output_size(str(self.orig_mrc), str(edf_path), self.params['tomo_binning'])
        print(f"   -> Rotation-aware size for newstack: {final_size}")
        
        newst_lines = (self.work_dir / "newst.com").read_text().splitlines()
        found_size = False
        inserted_bin = False
        
        for i, line in enumerate(newst_lines):
            if line.startswith("$newstack -StandardInput") and not inserted_bin:
                newst_lines.insert(i + 1, f"BinByFactor    {self.params['tomo_binning']}")
                inserted_bin = True
            elif line.startswith("SizeToOutputInXandY"):
                newst_lines[i] = f"SizeToOutputInXandY    {final_size}"
                found_size = True
                
        if not found_size:
            newst_lines.append(f"SizeToOutputInXandY    {final_size}")
            
        (self.work_dir / "newst.com").write_text("\n".join(newst_lines))
        run_cmd(["submfg", "newst.com"], cwd=self.work_dir)
        
        self._update_com("tilt.com", "IMAGEBINNED", str(self.params['tomo_binning']))
        self._update_com("tilt.com", "OutputFile", f"{self.base_name}_full_rec.mrc")
        self._update_com("tilt.com", "THICKNESS", str(self.params['final_thickness_px']))
        self._update_com("tilt.com", "XAXISTILT", "0")
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
    parser.add_argument("--template", type=str, default="krios.adoc", help="Path to IMOD system template (.adoc). Uses default 'krios.adoc' in tmeplates")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for python masking, default 8")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and plotting for the masking phase. Writes a useful histogram plot.")
    parser.add_argument("--skip_mask", action="store_true", help="Completely bypass outlier masking phase")
    
    # Advanced Masking Constraints
    parser.add_argument("--mask_low_cut", type=float, default=0.05, help="Lower threshold cut factor. Lower values (e.g. 0.05) result in gentler masking of dark features.")
    parser.add_argument("--mask_high_cut", type=float, default=0.05, help="Upper threshold cut factor. Lower values result in gentler masking of bright features/vacuum.")
    parser.add_argument("--mask_dilation", type=int, default=5, help="Mask boundary dilation in pixels. Lower values (e.g. 3) reduce masking spill-over into biological material. Default 5")
    parser.add_argument("--wiggle", type=float, default=1.0, help="Masking becomes more stringent with increasing tilt. Higher values make higher tilts be masked less harshly, default 1.")
    
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
                'aretomo_binning': args.aretomo_binning,
                'tomo_binning': args.tomo_binning,
                'imagebinned': args.imagebinned,
                'template_path': template_path,
                'workers': args.workers
            }
            if args.engine == 'aretomo2':
                pipeline = AreTomoEngine(input_mrc, work_dir, params)
            else:
                pipeline = EtomoEngine(input_mrc, work_dir, params)
            pipeline.run()
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed for {input_mrc.name}: {e}")
            continue

if __name__ == "__main__":
    main()