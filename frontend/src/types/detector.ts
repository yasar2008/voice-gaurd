export interface SignalContribution {
  raw_score: number;
  weight: number;
  contribution: number;
  interpretation: string;
}

export interface ProsodyDetail {
  pitch_mean: number;
  pitch_std: number;
  jitter: number;
  shimmer: number;
  hnr: number;
  pause_ratio: number;
  speaking_rate?: number;
}

export interface RiskAnalysis {
  risk_score: number;
  alert: boolean;
  confidence: 'low' | 'medium' | 'high';
  breakdown: {
    spoof_detection?: SignalContribution;
    speaker_similarity?: SignalContribution;
    prosody_naturalness?: SignalContribution;
    [key: string]: SignalContribution | undefined;
  };
  prosody_detail?: ProsodyDetail;
  anomalies: string[];
  latency_ms?: number;
  /** False when the spoof detector is running on uncalibrated weights. */
  calibrated?: boolean;
  quality?: AudioQuality;
  buffer_duration_s?: number;
  timestamp?: number;
}

/** Whether the input is inside the detector's measured operating envelope. */
export interface AudioQuality {
  reliable: boolean;
  reasons: string[];
  snr_db: number;
  high_freq_db: number;
  duration_s: number;
}

export interface SystemHealth {
  status: string;
  models_loaded: {
    spoof_detector: boolean;
    speaker_verifier: boolean;
    prosody_analyzer: boolean;
    risk_scorer: boolean;
  };
  device: string;
  enrolled_speaker: boolean;
  spoof_detector?: {
    backend: string;
    calibrated: boolean;
    checkpoint: string | null;
  };
}

export interface ConfigData {
  fusion_weights: {
    spoof_detection: number;
    speaker_similarity: number;
    prosody_naturalness: number;
  };
  alert_threshold: number;
  prosody_ranges: {
    jitter: [number, number];
    shimmer: [number, number];
    pitch_std: [number, number];
    hnr: [number, number];
    pause_ratio: [number, number];
  };
}
