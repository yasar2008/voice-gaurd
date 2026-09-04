"""
Generate synthetic speech samples for calibration, using the TTS voices that are
already installed on this machine (Windows SAPI via System.Speech).

No downloads and no cloud calls — but note what this does and does not prove:
SAPI's desktop voices are older concatenative/parametric engines, so they are an
*easy* spoof class. Passing here means the detector catches obvious TTS; it says
nothing about modern neural cloning (VITS, HiFi-GAN, ElevenLabs-class systems),
which is the actual threat model. Treat this as a floor, not a benchmark.

Usage:
    python scripts/generate_tts_samples.py                     # default en-US set
    python scripts/generate_tts_samples.py --out data/eval/spoof
    python scripts/generate_tts_samples.py --list-voices
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "eval" / "spoof_tts"

# Each block is read as one utterance. Long enough that every clip clears the
# ~4.04 s AASIST window with room to spare.
PASSAGES = [
    "Good morning. I am calling about the transfer that was scheduled for this "
    "afternoon. Could you please confirm whether the payment has already been "
    "released, or whether it is still waiting for approval?",

    "The quick brown fox jumps over the lazy dog. She sells seashells by the "
    "seashore. Peter Piper picked a peck of pickled peppers, and the rain in "
    "Spain falls mainly on the plain.",

    "I need you to listen carefully, because this is important. There has been "
    "unusual activity on the account, and we have to verify a few details before "
    "anything else can happen today.",

    "Weather for the region remains unsettled through the weekend, with scattered "
    "showers in the morning and clearing skies by late afternoon. Temperatures "
    "will stay close to the seasonal average.",

    "Thank you for holding. Your call is important to us. An agent will be with "
    "you shortly. Please have your reference number and date of birth ready when "
    "the call connects.",
]

# (voice name fragment, speaking rate) — rate varies the prosody a little so the
# set is not five clips of the identical cadence.
VARIANTS = [
    ("David", 0),
    ("Zira", 0),
    ("David", -2),
    ("Zira", 2),
]

PS_SCRIPT = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -like '*{voice}*' }} | Select-Object -First 1
if ($null -eq $voice) {{ Write-Error 'voice not found: {voice}'; exit 1 }}
$synth.SelectVoice($voice.VoiceInfo.Name)
$synth.Rate = {rate}
$synth.SetOutputToWaveFile('{path}')
$synth.Speak(@'
{text}
'@)
$synth.Dispose()
"""


def list_voices() -> int:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() "
        "| ForEach-Object { $_.VoiceInfo.Name + '  [' + $_.VoiceInfo.Culture + ']' }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode


def synthesize(text: str, voice: str, rate: int, path: Path) -> bool:
    script = PS_SCRIPT.format(voice=voice, rate=rate, path=str(path), text=text)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not path.exists():
        print(f"[fail] {path.name}: {result.stderr.strip()}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--list-voices", action="store_true", help="show installed voices and exit")
    args = parser.parse_args()

    if args.list_voices:
        return list_voices()

    if sys.platform != "win32":
        print("[fail] This script uses Windows SAPI. On Linux/macOS use a local TTS engine instead.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    made = 0

    for i, text in enumerate(PASSAGES):
        voice, rate = VARIANTS[i % len(VARIANTS)]
        name = f"tts_{voice.lower()}_r{rate:+d}_{i:02d}.wav"
        path = args.out / name
        # SAPI writes 16-bit PCM WAV; the pipeline resamples to 16 kHz itself.
        if synthesize(text.replace("'", "''"), voice, rate, path):
            size_kb = path.stat().st_size / 1024
            print(f"[ok  ] {name} ({size_kb:.0f} KB)")
            made += 1

    print(f"\n{made}/{len(PASSAGES)} synthetic samples written to {args.out}")
    print("Score them with:")
    print(f"    python scripts/calibrate.py --spoof {args.out}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
