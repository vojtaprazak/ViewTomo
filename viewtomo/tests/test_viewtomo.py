import unittest
import os
import tempfile
import subprocess
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

# Suppress noise from third-party dependencies (e.g., matplotlib's pyparsing deprecations)
warnings.filterwarnings("ignore", module=".*matplotlib.*")
warnings.filterwarnings("ignore", module=".*pyparsing.*")

# Attempt relative import assuming the tests folder is adjacent to viewtomo folder
try:
    from viewtomo.tomo_utils import (
        calculate_imod_binning,
        resolve_template_path,
        append_or_replace_adoc_keys,
        calculate_thicknesses,
        run_cmd,
        determine_output_size
    )
except ImportError:
    pass # In production environments, rely on standard site-packages

class TestTomoUtils(unittest.TestCase):

    def test_calculate_imod_binning(self):
        """Tests the exact parity adjustment logic for IMOD newstack boundaries."""
        
        # Test 1: Even original dimension, odd standard binning
        # 4092 // 4 = 1023 (odd). 4092 is even. Parity mismatch!
        # nbin_out = 1024, irem = 4. ix_offset = -(4 - 2) = -2
        nbin_out, ix_offset = calculate_imod_binning(4092, 4)
        self.assertEqual(nbin_out, 1024)
        self.assertEqual(ix_offset, -2)

        # Test 2: Even original dimension, even standard binning
        # 5760 // 4 = 1440 (even). 5760 is even. Match!
        # irem = 0. No offset.
        nbin_out, ix_offset = calculate_imod_binning(5760, 4)
        self.assertEqual(nbin_out, 1440)
        self.assertEqual(ix_offset, 0)

    def test_resolve_template_path(self):
        """Tests resolution logic using a temporary dummy file."""
        with tempfile.NamedTemporaryFile(suffix=".adoc", delete=False) as tf:
            dummy_path = tf.name
            
        try:
            resolved = resolve_template_path(dummy_path)
            self.assertEqual(str(resolved), str(Path(dummy_path).resolve()))
        finally:
            os.remove(dummy_path)

    def test_append_or_replace_adoc_keys(self):
        """Tests regex-based configuration file modifications."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".adoc", delete=False) as tf:
            tf.write("setupset.copyarg.name=OldName\n")
            tf.write("some.other.key=42\n")
            dummy_path = tf.name
            
        try:
            overrides = {
                "setupset.copyarg.name": "NewName",     # Replace existing
                "setupset.copyarg.gold": "10.0"         # Append new
            }
            
            append_or_replace_adoc_keys(Path(dummy_path), overrides)
            
            with open(dummy_path, 'r') as f:
                content = f.read()
                
            self.assertIn("setupset.copyarg.name=NewName", content)
            self.assertNotIn("OldName", content)
            self.assertIn("setupset.copyarg.gold=10.0", content)
        finally:
            os.remove(dummy_path)

    @patch("viewtomo.tomo_utils.mrcfile.open")
    def test_calculate_thicknesses(self, mock_mrc_open):
        """Tests pixel thickness calculations dynamically determined from an MRC header."""
        mock_mrc = MagicMock()
        mock_mrc.voxel_size.x = 1.5  # 1.5 Angstroms/px
        mock_mrc_open.return_value.__enter__.return_value = mock_mrc

        # Align target: 150 nm = 1500 A -> 1500 / 1.5 = 1000 px (even)
        # Final target: 300 nm = 3000 A -> 3000 / 1.5 = 2000 px (even)
        result = calculate_thicknesses("dummy.mrc", 150.0, 300.0)

        self.assertEqual(result["apix_angstroms"], 1.5)
        self.assertEqual(result["align_thickness_px"], 1000)
        self.assertEqual(result["final_thickness_px"], 2000)

    @patch("viewtomo.tomo_utils.subprocess.run")
    def test_run_cmd_success(self, mock_run):
        """Tests that run_cmd properly executes without error."""
        mock_run.return_value.returncode = 0
        
        # Should not raise an exception
        run_cmd(["echo", "hello"], cwd="/tmp")
        mock_run.assert_called_once()

    @patch("viewtomo.tomo_utils.subprocess.run")
    def test_run_cmd_failure(self, mock_run):
        """Tests that run_cmd properly raises a RuntimeError on failure to safeguard batch processing."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "fake_cmd")
        
        with self.assertRaises(RuntimeError):
            run_cmd(["fake_cmd"], cwd="/tmp")

    @patch("viewtomo.tomo_utils.subprocess.run")
    def test_determine_output_size(self, mock_run):
        """Tests output size calculation with logic for 90-degree image rotations."""
        # Mock IMOD header output (nx, ny, nz)
        mock_run.return_value.stdout = "4000 3000 100\n"
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".edf", delete=False) as tf:
            # A 90 degree rotation means nx and ny should be swapped
            tf.write("ImageRotationA=90.0\n")
            dummy_edf = tf.name
            
        try:
            # Swapped dims: nx=3000, ny=4000. Binning by 4 -> 750, 1000. 
            result = determine_output_size("dummy.mrc", dummy_edf, 4)
            self.assertEqual(result, "750,1000")
        finally:
            os.remove(dummy_edf)

if __name__ == "__main__":
    unittest.main()