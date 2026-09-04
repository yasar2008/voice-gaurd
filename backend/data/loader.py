"""
ASVspoof 2019 LA Dataset Loader.

Parses the ASVspoof 2019 Logical Access protocol files and provides:
- A PyTorch Dataset for training/evaluation
- Utility functions for loading individual samples
- Protocol file parsing (bonafide vs. spoof labels, attack types)

Dataset structure expected:
    data/
    └── ASVspoof2019/
        └── LA/
            ├── ASVspoof2019_LA_train/flac/
            ├── ASVspoof2019_LA_dev/flac/
            ├── ASVspoof2019_LA_eval/flac/
            └── ASVspoof2019_LA_cm_protocols/
                ├── ASVspoof2019.LA.cm.train.trn.txt
                ├── ASVspoof2019.LA.cm.dev.trl.txt
                └── ASVspoof2019.LA.cm.eval.trl.txt

Protocol file format (space-separated):
    SPEAKER_ID AUDIO_FILE_ID - ATTACK_TYPE LABEL
    Example: LA_0079 LA_T_1138215 - A04 spoof
    Example: LA_0079 LA_T_1138296 - - bonafide
"""

from pathlib import Path
from typing import Optional

import torch
import torchaudio

from backend.data.audio_io import load_audio
from torch.utils.data import Dataset

from backend.config import config, PROJECT_ROOT


# Label mapping
LABEL_MAP = {"bonafide": 0, "spoof": 1}
LABEL_MAP_INV = {0: "bonafide", 1: "spoof"}


def parse_protocol_file(protocol_path: str | Path) -> list[dict]:
    """
    Parse an ASVspoof protocol file.
    
    Args:
        protocol_path: Path to the protocol text file
        
    Returns:
        List of dicts with keys: speaker_id, audio_id, attack_type, label
    """
    entries = []
    protocol_path = Path(protocol_path)

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            speaker_id = parts[0]
            audio_id = parts[1]
            # parts[2] is always "-"
            attack_type = parts[3]  # "A01"–"A19" for spoof, "-" for bonafide
            label = parts[4] if len(parts) > 4 else ("bonafide" if attack_type == "-" else "spoof")

            entries.append(
                {
                    "speaker_id": speaker_id,
                    "audio_id": audio_id,
                    "attack_type": attack_type,
                    "label": label,
                    "label_int": LABEL_MAP.get(label, 1),
                }
            )

    return entries


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 LA.
    
    Args:
        data_root: Root directory containing the ASVspoof2019 data
        split: One of "train", "dev", "eval"
        max_length: Maximum waveform length in samples (pad/truncate)
        transform: Optional transform to apply to waveforms
    """

    SPLIT_CONFIG = {
        "train": {
            "audio_dir": "ASVspoof2019_LA_train/flac",
            "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        },
        "dev": {
            "audio_dir": "ASVspoof2019_LA_dev/flac",
            "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        },
        "eval": {
            "audio_dir": "ASVspoof2019_LA_eval/flac",
            "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        },
    }

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        max_length: Optional[int] = None,
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_length = max_length or config.model.aasist_input_length
        self.transform = transform
        self.target_sr = config.audio.sample_rate

        if split not in self.SPLIT_CONFIG:
            raise ValueError(f"Invalid split '{split}'. Choose from: {list(self.SPLIT_CONFIG)}")

        split_cfg = self.SPLIT_CONFIG[split]
        self.audio_dir = self.data_root / split_cfg["audio_dir"]
        protocol_path = self.data_root / split_cfg["protocol"]

        # Parse protocol file
        self.entries = parse_protocol_file(protocol_path)

        # Verify audio directory exists
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found: {self.audio_dir}. "
                f"Download ASVspoof 2019 LA from https://datashare.ed.ac.uk/handle/10283/3336"
            )

        print(
            f"[ASVspoofDataset] Loaded {split} split: {len(self.entries)} entries "
            f"({sum(1 for e in self.entries if e['label'] == 'bonafide')} bonafide, "
            f"{sum(1 for e in self.entries if e['label'] == 'spoof')} spoof)"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        audio_path = self.audio_dir / f"{entry['audio_id']}.flac"

        # Load audio
        waveform, sr = load_audio(str(audio_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Pad or truncate
        waveform = self._pad_or_truncate(waveform)

        # Apply optional transform
        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform.squeeze(0),  # [max_length]
            "label": entry["label_int"],
            "speaker_id": entry["speaker_id"],
            "audio_id": entry["audio_id"],
            "attack_type": entry["attack_type"],
        }

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad with zeros or truncate to self.max_length."""
        length = waveform.shape[1]

        if length < self.max_length:
            pad = self.max_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif length > self.max_length:
            # Random crop during training, center crop otherwise
            if self.split == "train":
                start = torch.randint(0, length - self.max_length, (1,)).item()
            else:
                start = (length - self.max_length) // 2
            waveform = waveform[:, start : start + self.max_length]

        return waveform

    def get_label_counts(self) -> dict[str, int]:
        """Return count of bonafide and spoof samples."""
        counts = {"bonafide": 0, "spoof": 0}
        for entry in self.entries:
            counts[entry["label"]] += 1
        return counts

    def get_attack_type_counts(self) -> dict[str, int]:
        """Return count of each attack type."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            at = entry["attack_type"]
            counts[at] = counts.get(at, 0) + 1
        return counts
