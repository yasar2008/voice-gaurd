export type VerdictKey = 'idle' | 'real' | 'unsure' | 'synthetic';

export interface Verdict {
  key: VerdictKey;
  /** Headline shown under the listen button. */
  label: string;
  /** One-line plain-English explanation. */
  detail: string;
  color: string;
  glow: string;
}

const VERDICTS: Record<VerdictKey, Omit<Verdict, 'key'>> = {
  idle: {
    label: 'Ready',
    detail: 'Tap to listen to the voice around you.',
    color: 'var(--muted-foreground)',
    glow: 'rgba(161, 161, 170, 0.18)',
  },
  real: {
    label: 'Real voice',
    detail: 'Speech looks human — no synthesis artefacts found.',
    color: 'var(--emerald)',
    glow: 'rgba(16, 185, 129, 0.22)',
  },
  unsure: {
    label: 'Uncertain',
    detail: 'Mixed signals. Keep listening for a few more seconds.',
    color: 'var(--amber)',
    glow: 'rgba(245, 158, 11, 0.22)',
  },
  synthetic: {
    label: 'Synthetic voice',
    detail: 'This audio carries the fingerprints of a cloned or TTS voice.',
    color: 'var(--rose)',
    glow: 'rgba(244, 63, 94, 0.26)',
  },
};

/**
 * Map a fused risk score onto the three-state verdict the home screen shows.
 *
 * `score` should be the smoothed score and `alert` the hysteresis-gated flag,
 * so a single noisy window cannot flip the headline.
 */
export function verdictFor(
  score: number | null,
  alert: boolean,
  threshold: number,
  suspiciousAt = 35,
): Verdict {
  if (score === null) return { key: 'idle', ...VERDICTS.idle };

  const clamped = Math.max(0, Math.min(100, score));
  let key: VerdictKey = 'real';
  if (alert && clamped >= threshold) key = 'synthetic';
  else if (clamped >= suspiciousAt) key = 'unsure';

  return { key, ...VERDICTS[key] };
}
