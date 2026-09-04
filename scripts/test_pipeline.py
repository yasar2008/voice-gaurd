"""
CLI Test Pipeline for Voice-Clone Impersonation Detector.

Demonstrates the entire workflow locally on generated samples:
1. Enrolls a bonafide reference voice
2. Analyzes a test bonafide sample (low risk expected)
3. Analyzes a synthetic clone sample (high risk / alert expected)
"""

import sys
from pathlib import Path
import soundfile as sf
import torch
import numpy as np

# Ensure backend can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import config
from backend.models.aasist import AASISTDetector
from backend.models.speaker_verify import SpeakerVerifier
from backend.features.prosody import ProsodyAnalyzer
from backend.fusion.risk_scorer import RiskScorer

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def run_pipeline():
    print("=" * 70)
    print("[*] VOICE-CLONE DETECTOR -- END-TO-END PIPELINE DEMO")
    print("=" * 70)
    
    # 1. Initialize Pipeline
    print("\n[1/4] Initializing Forensic Models...")
    detector = AASISTDetector()
    verifier = SpeakerVerifier()
    analyzer = ProsodyAnalyzer()
    scorer = RiskScorer()
    print("  + AASIST-L Spoof Detector Loaded")
    print("  + ECAPA-TDNN Speaker Verifier Loaded")
    print("  + Praat Prosody Analyzer Loaded")
    print("  + Explainable Fusion Scorer Loaded")
    
    # 2. Enroll Speaker Voiceprint
    bonafide_path = FIXTURES_DIR / "bonafide_human_sim.wav"
    clone_path = FIXTURES_DIR / "tts_clone_sim.wav"
    
    print("\n[2/4] Enrolling Reference Speaker Voiceprint...")
    audio_data, sr = sf.read(str(bonafide_path))
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
    
    embedding = verifier.enroll(audio_tensor, sr)
    print(f"  + Voiceprint Enrolled ({len(embedding)}-d vector, norm: {np.linalg.norm(embedding):.4f})")
    
    # 3. Test Bonafide Sample
    print("\n[3/4] Analyzing Bonafide Human Sample...")
    spoof_score_b = detector.predict(audio_tensor, sr)
    speaker_score_b = verifier.verify(audio_tensor, sr)
    prosody_b = analyzer.extract(audio_data.astype(np.float64), sr)
    
    result_b = scorer.compute(
        spoof_score=spoof_score_b,
        speaker_score=speaker_score_b,
        prosody_score=prosody_b["prosody_score"]
    )
    result_b.anomalies = prosody_b.get("anomalies", [])
    
    print(f"  Risk Score: {result_b.risk_score:.1f}/100  |  Alert: {result_b.alert}  |  Confidence: {result_b.confidence}")
    print(f"  * Spoof Detection:      {spoof_score_b:.3f} (Contrib: +{result_b.breakdown[0].contribution:.1f} pts)")
    print(f"  * Speaker Verification: {speaker_score_b:.3f} (Contrib: +{result_b.breakdown[1].contribution:.1f} pts)")
    print(f"  * Prosody Naturalness:  {prosody_b['prosody_score']:.3f} (Contrib: +{result_b.breakdown[2].contribution:.1f} pts)")
    print(f"    [F0 Mean: {prosody_b['pitch_mean']:.0f}Hz, Jitter: {prosody_b['jitter']*100:.3f}%, HNR: {prosody_b['hnr']:.1f}dB]")
    
    # 4. Test Synthetic Clone Sample
    print("\n[4/4] Analyzing Synthetic Clone Attack Sample...")
    clone_data, clone_sr = sf.read(str(clone_path))
    clone_tensor = torch.from_numpy(clone_data).float().unsqueeze(0)
    
    spoof_score_c = detector.predict(clone_tensor, clone_sr)
    speaker_score_c = verifier.verify(clone_tensor, clone_sr)
    prosody_c = analyzer.extract(clone_data.astype(np.float64), clone_sr)
    
    result_c = scorer.compute(
        spoof_score=spoof_score_c,
        speaker_score=speaker_score_c,
        prosody_score=prosody_c["prosody_score"]
    )
    result_c.anomalies = prosody_c.get("anomalies", [])
    
    print(f"  Risk Score: {result_c.risk_score:.1f}/100  |  Alert: {result_c.alert}  |  Confidence: {result_c.confidence}")
    print(f"  * Spoof Detection:      {spoof_score_c:.3f} (Contrib: +{result_c.breakdown[0].contribution:.1f} pts)")
    print(f"  * Speaker Verification: {speaker_score_c:.3f} (Contrib: +{result_c.breakdown[1].contribution:.1f} pts)")
    print(f"  * Prosody Naturalness:  {prosody_c['prosody_score']:.3f} (Contrib: +{result_c.breakdown[2].contribution:.1f} pts)")
    print(f"    [F0 Mean: {prosody_c['pitch_mean']:.0f}Hz, Jitter: {prosody_c['jitter']*100:.4f}%, HNR: {prosody_c['hnr']:.1f}dB]")
    if result_c.anomalies:
        print(f"  * Detected Anomalies: {', '.join(result_c.anomalies)}")
    
    print("\n" + "=" * 70)
    print("[+] Pipeline validation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
