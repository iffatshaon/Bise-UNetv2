import unittest
import numpy as np
import cv2
import sys
import os

# Add parent to path to find postprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocess.eval_utils import calculate_hd95_asd

class TestMetrics(unittest.TestCase):
    def test_perfect_match(self):
        # Two identical squares
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        hd95, asd = calculate_hd95_asd(mask, mask)
        self.assertEqual(hd95, 0.0)
        self.assertEqual(asd, 0.0)
        
    def test_empty_masks(self):
        # Both empty
        mask = np.zeros((100, 100), dtype=np.uint8)
        hd95, asd = calculate_hd95_asd(mask, mask)
        self.assertEqual(hd95, 0.0)
        self.assertEqual(asd, 0.0)
        
    def test_one_empty(self):
        # One empty
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m2[10:20, 10:20] = 1
        
        hd95, asd = calculate_hd95_asd(m1, m2)
        self.assertTrue(np.isnan(hd95))
        self.assertTrue(np.isnan(asd))
        
    def test_translation(self):
        # 10 pixel shift
        # 100x100 image
        # Square 1: x=20..80 (width 60)
        # Square 2: x=30..90 (width 60)
        # Vertical is same y=20..80
        
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m1[20:80, 20:80] = 1
        
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m2[20:80, 30:90] = 1
        
        # Distance logic:
        # Left edge of m1 (x=20) is 10px from Left edge of m2 (x=30)??
        # Nearest boundary of m1 points to m2 boundary.
        # This is complex to calculate mentally exact, but max distance should be around 10.
        # HD95 should be close to 10.
        
        hd95, asd = calculate_hd95_asd(m1, m2)
        
        # Max distance is clearly 10 (the shift).
        # So HD95 should be <= 10.
        # ASD should be <= 10.
        # And it shouldn't be 0
        self.assertTrue(0 < hd95 <= 10.0 + 1e-6) # float tol
        self.assertTrue(0 < asd <= 10.0 + 1e-6)
        print(f"Translation Test: HD95={hd95:.4f}, ASD={asd:.4f}")

if __name__ == "__main__":
    unittest.main()
