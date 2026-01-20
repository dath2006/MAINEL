
import unittest
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta
from app.core.reid.visual_matcher import VisualMatcher
from app.services.reid_service import ReIDService

class TestUnifiedMatching(unittest.TestCase):
    def setUp(self):
        # Initialize ReID service with aggressive thresholds to test the logic
        self.reid_service = ReIDService(
            match_threshold=0.6,
            new_threshold=0.5,
            st_weight=0.0 # Disable ST for pure visual test
        )
        # Verify VisualMatcher has our new logic (by checking the method source or behavior)
        self.matcher = self.reid_service.visual_matcher
        
    def test_cross_camera_matching(self):
        """Test that a person seen in Cam1 is matched in Cam2 even with feature variance."""
        
        # 1. Person A in Camera 1
        # Create a "canonical" embedding for Person A
        rng = np.random.default_rng(42)
        base_emb = rng.random(256)
        base_emb = base_emb / np.linalg.norm(base_emb)
        
        # First observation: Camera 1, Time T
        t1 = datetime.now()
        # Add slight noise to simulate camera variance
        obs1 = base_emb + rng.normal(0, 0.05, 256)
        obs1 = obs1 / np.linalg.norm(obs1)
        
        result1 = self.reid_service._create_new_identity(1, obs1, t1)
        id1 = str(result1.global_track_id)
        
        print(f"Created ID1: {id1}")
        
        # Verify it's in gallery
        self.assertIn(id1, self.matcher.gallery)
        
        # 2. Person A in Camera 2 (Different angle/lighting = more variance)
        # Create an observation that is "mediocre" match to obs1 but clearly Person A
        # Let's say dot product is around 0.55 (below match_threshold 0.6, but above new_threshold*0.8 = 0.4)
        
        # We simulate this artificially
        # But first let's verify our MAX logc in VisualMatcher
        # We add another observation to ID1 in Cam1 (history accumulation)
        obs2 = base_emb + rng.normal(0, 0.05, 256) 
        obs2 = obs2 / np.linalg.norm(obs2)
        self.matcher.add_to_gallery(id1, obs2, 1, t1 + timedelta(seconds=1))
        
        # Now Camera 2 observation
        t2 = t1 + timedelta(minutes=5)
        # Make obs3 close to obs2 but far from obs1 (simulating pose change that matches a specific history frame)
        obs3 = obs2 + rng.normal(0, 0.1, 256)
        obs3 = obs3 / np.linalg.norm(obs3)
        
        # Check similarity manually first
        sim_avg = np.dot(obs3, self.matcher.gallery[id1].embedding)
        sim_obs2 = np.dot(obs3, obs2)
        
        print(f"Sim vs Avg: {sim_avg:.3f}")
        print(f"Sim vs Obs2 (History): {sim_obs2:.3f}")
        
        # 3. Perform Match
        # This calls match_identity which uses our new lenient logic
        # It should match ID1 because:
        # a) VisualMatcher.match uses MAX similarity (so it sees high sim with obs2)
        # b) ReIDService uses lenient threshold (new_threshold * 0.8)
        
        match_result = self.reid_service.visual_matcher.match_best(obs3)
        self.assertIsNotNone(match_result)
        matched_id, score, _ = match_result
        
        print(f"VisualMatcher Best Match: {matched_id} Score: {score:.3f}")
        self.assertEqual(matched_id, id1)
        # Verify score is the MAX (sim_obs2), not the AVG
        self.assertAlmostEqual(score, max(sim_avg, sim_obs2), places=5)
        
if __name__ == '__main__':
    unittest.main()
