import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from h2o_dataset import H2ODataset


DATA_ROOT = Path("h2o_data")


@unittest.skipUnless(DATA_ROOT.is_dir(), "H2O source data is not mounted")
class DatasetIntegrationTest(unittest.TestCase):
    def test_fixed_single_and_dataloader(self):
        dataset = H2ODataset(
            robot_id="unitree_g1",
            split="test",
            clip_length=3,
            window_stride=1000000,
            view_mode="fixed_single",
            image_size=(64, 64),
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["visual"]["rgb"].shape), (1, 3, 3, 64, 64))
        self.assertEqual(tuple(sample["visual"]["depth"].shape), (1, 3, 1, 64, 64))
        self.assertEqual(sample["visual"]["depth"].dtype, torch.float32)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(batch["visual"]["rgb"].shape), (2, 1, 3, 3, 64, 64))

    def test_all_views_background_is_deterministic(self):
        dataset = H2ODataset(
            robot_id="unitree_g1",
            split="test",
            clip_length=2,
            window_stride=1000000,
            view_mode="all",
            image_size=(64, 64),
            use_background=True,
            background_probability=1.0,
        )
        first = dataset[0]
        second = dataset[0]
        self.assertEqual(tuple(first["visual"]["rgb"].shape), (4, 2, 3, 64, 64))
        self.assertTrue(torch.equal(first["visual"]["rgb"], second["visual"]["rgb"]))
        self.assertEqual(first["background"]["ids"], second["background"]["ids"])


if __name__ == "__main__":
    unittest.main()
