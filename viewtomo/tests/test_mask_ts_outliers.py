import unittest
import numpy as np
import logging
import sys

try:
    from viewtomo.mask_ts_outliers import PhysicsModel, ImageProcessor
except ImportError:
    pass # Expected environment layout during pytest run

class TestMaskTSOutliers(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy image mimicking a binned tomogram slice
        self.dummy_image = np.random.rand(100, 100) * 100
        # Add a fake high-intensity (vacuum) region
        self.dummy_image[40:60, 40:60] = 1000 
        
        # Setup a dummy logger for calculate_thresholds
        self.dummy_logger = logging.getLogger("TestLogger")
        self.dummy_logger.setLevel(logging.CRITICAL)

    def test_robust_stats(self):
        """Tests median and IQR calculations on the image center."""
        med, iqr = ImageProcessor.robust_stats(self.dummy_image)
        self.assertTrue(med > 0)
        self.assertTrue(iqr >= 0)

    def test_bin_ndarray(self):
        """Verifies safe block-averaging reduction for large 3D stacks."""
        data = np.ones((5, 64, 64))
        binned = PhysicsModel.bin_ndarray(data, binning=2)
        # Expected: Z remains 5, X/Y divided by 2
        self.assertEqual(binned.shape, (5, 32, 32))

    def test_estimate_tilt_offset(self):
        """Validates robust parabolic fitting for tilt center estimation."""
        tilts = np.linspace(-60, 60, 41)
        
        # Create a fake, perfectly centered bell curve offset by +10 degrees
        intensities = 1000 * np.cos(np.radians(tilts - 10))
        intensities[intensities < 0] = 0.1 # Prevent neg log
        
        offset = PhysicsModel.estimate_tilt_offset(tilts, intensities)
        
        # Should be able to detect the peak is at 10.0 degrees
        self.assertAlmostEqual(offset, 10.0, delta=2.0)

    def test_generate_mask(self):
        """Checks if boundary thresholding combined with edge logic generates a valid bool mask."""
        mask = ImageProcessor.generate_mask(
            self.dummy_image, low_cut=10, high_cut=900, 
            med=50, iqr=20, sigma=1.0, dilation=2, dust_threshold=0
        )
        self.assertEqual(mask.shape, self.dummy_image.shape)
        self.assertEqual(mask.dtype, bool)
        
        # Ensure our massive fake high-intensity spot triggered the high_cut
        self.assertTrue(mask[50, 50])

    def test_inpaint_slice(self):
        """Ensures noise generation and blending runs successfully without altering shape."""
        mask = np.zeros_like(self.dummy_image, dtype=bool)
        mask[40:60, 40:60] = True
        
        inpainted = ImageProcessor.inpaint_slice(self.dummy_image, mask, soft_sigma=2.0)
        
        self.assertEqual(inpainted.shape, self.dummy_image.shape)
        self.assertEqual(inpainted.dtype, self.dummy_image.dtype)
        
        # The extreme 1000 value should have been replaced with ~median noise
        self.assertTrue(inpainted[50, 50] < 500)

    def test_calculate_thresholds_returns_arrays(self):
        """Checks if the physics fit gracefully handles an ideal dataset."""
        tilts = np.linspace(-60, 60, 11)
        data = np.zeros((11, 32, 32))
        
        for i, t in enumerate(tilts):
            # Perfect cosine decay mimicking Beer Lambert sample attenuation
            data[i, :, :] = 1000 * np.cos(np.radians(t)) + np.random.rand(32, 32)
        
        low, high, success = PhysicsModel.calculate_thresholds(
            self.dummy_logger, data, n_tilts=11, binning=1
        )
        
        self.assertEqual(len(low), 11)
        self.assertEqual(len(high), 11)
        
if __name__ == "__main__":
    unittest.main()