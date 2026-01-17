import os
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from .config import Config
    from .transforms import get_train_transforms, get_val_transforms, get_degradation_transforms
except ImportError:
    from config import Config
    from transforms import get_train_transforms, get_val_transforms, get_degradation_transforms


class AdvancedMultiFrameDataset(Dataset):
    """
    Multi-frame license plate dataset.
    
    Loads tracks containing multiple LR/HR frames and their annotations.
    For training, randomly switches between LR images and HR+degradation.
    """
    
    def __init__(
        self,
        root_dir,
        mode='train',
        split_ratio=0.9,
        test_ratio=0.0,
        hr_plus_root: str = "",
        hr_plus_method: str = "lanczos2",
        hr_plus_scale: int = 4,
    ):
        """
        Args:
            root_dir: Path to data directory containing track_* folders
            mode: 'train' or 'val'
            split_ratio: Ratio of data to use for training
        """
        self.mode = mode
        self.split_ratio = float(split_ratio)
        self.test_ratio = float(test_ratio)
        self.hr_plus_root = str(hr_plus_root or "")
        self.hr_plus_method = str(hr_plus_method or "lanczos2")
        self.hr_plus_scale = int(hr_plus_scale)
        self.samples = []
        
        if mode == 'train':
            self.transform = get_train_transforms()
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms()
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        self.abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(self.abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ LỖI: Không tìm thấy data.")
            return

        train_tracks, val_tracks, test_tracks = self._split_tracks(all_tracks)
        
        if mode == 'train':
            selected_tracks = train_tracks
        elif mode == 'val':
            selected_tracks = val_tracks
        elif mode == 'test':
            selected_tracks = test_tracks
        else:
            raise ValueError("mode must be one of: 'train', 'val', 'test'")
        print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
        
        self._load_samples(selected_tracks)

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        """Normalize label text to match Config.CHARS.

        - Uppercase
        - Remove whitespace
        - Keep only characters present in Config.CHAR2IDX
        """
        if not raw_label:
            return ""
        normalized = "".join(raw_label.split()).upper()
        normalized = "".join([c for c in normalized if c in Config.CHAR2IDX])
        return normalized

    def _maybe_map_to_hr_plus(self, img_path: str) -> str:
        """Map an LR frame path to a precomputed HR_plus output path if available."""
        if not self.hr_plus_root:
            return img_path
        try:
            rel = Path(img_path).resolve().relative_to(Path(self.abs_root).resolve())
        except Exception:
            return img_path

        # HR_plus export writes: <hr_plus_root>/x<scale>/<method>/<rel.parent>/<lr_stem>_x<scale>.png
        out = (
            Path(self.hr_plus_root)
            / f"x{int(self.hr_plus_scale)}"
            / self.hr_plus_method
            / rel.parent
            / f"{Path(img_path).stem}_x{int(self.hr_plus_scale)}.png"
        )
        return str(out) if out.exists() else img_path
    
    def _split_tracks(self, all_tracks):
        """Split tracks into train/val(/test) sets, persisting to JSON for reproducibility."""
        train_tracks: list[str] = []
        val_tracks: list[str] = []
        test_tracks: list[str] = []

        want_test = self.test_ratio > 0
        val_exists = os.path.exists(Config.VAL_SPLIT_FILE)
        test_exists = os.path.exists(getattr(Config, "TEST_SPLIT_FILE", "")) if want_test else False

        # If we want a test split, require both split files; otherwise, create a fresh 3-way split.
        # This avoids silently changing only part of the split.
        should_load = val_exists and ((not want_test) or test_exists)

        if should_load:
            if want_test:
                print(f"📂 Loading splits from '{Config.VAL_SPLIT_FILE}' and '{Config.TEST_SPLIT_FILE}'...")
            else:
                print(f"📂 Loading split from '{Config.VAL_SPLIT_FILE}'...")

            try:
                with open(Config.VAL_SPLIT_FILE, 'r') as f:
                    val_ids = set(json.load(f))
                test_ids = set()
                if want_test:
                    with open(Config.TEST_SPLIT_FILE, 'r') as f:
                        test_ids = set(json.load(f))
            except:
                val_ids = set()
                test_ids = set()
                print("⚠️ Lỗi đọc file split, sẽ tạo lại.")
                should_load = False

            if should_load:
                for t in all_tracks:
                    track_name = os.path.basename(t)
                    if track_name in val_ids:
                        val_tracks.append(t)
                    elif want_test and track_name in test_ids:
                        test_tracks.append(t)
                    else:
                        train_tracks.append(t)

                # If splits don't match current data, recreate
                if (not val_tracks) or (want_test and not test_tracks):
                    print("⚠️ File split không khớp với dữ liệu hiện tại. Chia lại...")
                    should_load = False

        if not should_load:
            if want_test:
                print("⚠️ Creating new train/val/test split...")
                train_tracks, val_tracks, test_tracks = self._create_new_split_3way(all_tracks)
            else:
                print("⚠️ Creating new train/val split...")
                train_tracks, val_tracks = self._create_new_split_2way(all_tracks)
                test_tracks = []

        return train_tracks, val_tracks, test_tracks
    
    def _create_new_split_2way(self, all_tracks):
        """Create a new train/val split and save to JSON."""
        all_tracks = list(all_tracks)
        random.Random(Config.SEED).shuffle(all_tracks)
        split_idx = int(len(all_tracks) * self.split_ratio)
        train_tracks = all_tracks[:split_idx]
        val_tracks = all_tracks[split_idx:]

        val_ids = [os.path.basename(t) for t in val_tracks]
        with open(Config.VAL_SPLIT_FILE, 'w') as f:
            json.dump(val_ids, f, indent=2)

        return train_tracks, val_tracks

    def _create_new_split_3way(self, all_tracks):
        """Create a new train/val/test split and save to JSON."""
        if not getattr(Config, "TEST_SPLIT_FILE", ""):
            raise ValueError("Config.TEST_SPLIT_FILE must be set when using test_ratio")

        all_tracks = list(all_tracks)
        random.Random(Config.SEED).shuffle(all_tracks)

        if self.test_ratio < 0 or self.split_ratio <= 0 or (self.split_ratio + self.test_ratio) >= 1:
            raise ValueError("Invalid split ratios: require split_ratio > 0, test_ratio >= 0, split_ratio + test_ratio < 1")

        n = len(all_tracks)
        n_train = int(n * self.split_ratio)
        n_test = int(n * self.test_ratio)
        n_val = n - n_train - n_test

        train_tracks = all_tracks[:n_train]
        val_tracks = all_tracks[n_train : n_train + n_val]
        test_tracks = all_tracks[n_train + n_val :]

        val_ids = [os.path.basename(t) for t in val_tracks]
        test_ids = [os.path.basename(t) for t in test_tracks]
        with open(Config.VAL_SPLIT_FILE, 'w') as f:
            json.dump(val_ids, f, indent=2)
        with open(Config.TEST_SPLIT_FILE, 'w') as f:
            json.dump(test_ids, f, indent=2)

        return train_tracks, val_tracks, test_tracks
    
    def _load_samples(self, tracks):
        """Load sample metadata from track directories."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                
                raw_label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                label = self._normalize_label(raw_label)
                if not label:
                    continue

                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) + 
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) + 
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                if len(lr_files) > 0:
                    self.samples.append({
                        'lr_paths': lr_files,
                        'hr_paths': hr_files,
                        'label': label
                    })
            except:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        
        # During training, 50% chance to use HR images with degradation
        use_hr = (self.mode == 'train') and (len(item['hr_paths']) > 0) and (random.random() < 0.5)
        
        if use_hr:
            images_list = self._load_frames(item['hr_paths'], apply_degradation=True)
        else:
            images_list = self._load_frames(item['lr_paths'], apply_degradation=False)

        images_tensor = torch.stack(images_list, dim=0)
        target = [Config.CHAR2IDX[c] for c in label]
            
        return images_tensor, torch.tensor(target, dtype=torch.long), len(target), label
    
    def _load_frames(self, paths, apply_degradation=False):
        """Load and process frames, padding to 5 frames if needed."""
        # Ensure exactly 5 frames
        if len(paths) < 5:
            paths = paths + [paths[-1]] * (5 - len(paths))
        else:
            paths = paths[:5]
        
        images_list = []
        for p in paths:
            # Use HR_plus only for LR branch (apply_degradation=False).
            if not apply_degradation:
                p = self._maybe_map_to_hr_plus(p)
            image = cv2.imread(p)
            if image is None:
                image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if apply_degradation and self.degrade:
                image = self.degrade(image=image)['image']
            
            image = self.transform(image=image)['image']
            images_list.append(image)
        
        return images_list

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        images, targets, target_lengths, labels_text = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text
