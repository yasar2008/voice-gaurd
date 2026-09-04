'use client';

import React from 'react';
import {
  Activity,
  ArrowRight,
  Sparkles,
  Settings2,
  BookOpen,
  Radio,
  Cpu,
  Globe2,
  Layers,
} from 'lucide-react';
import GlobeStudy from '@/components/ui/globe-study';

interface LandingViewProps {
  onLaunchDetector: () => void;
  onOpenSettings: () => void;
  onOpenAbout: () => void;
}

export const LandingView: React.FC<LandingViewProps> = ({
  onLaunchDetector,
  onOpenSettings,
  onOpenAbout,
}) => {
  return (
    <div className="landing-container">
      {/* Minimal Hero Header */}
      <section className="landing-hero">
        <div className="landing-badge">
          <Sparkles size={13} className="landing-badge-icon" />
          <span>Real-Time Voice Impersonation Defense</span>
        </div>

        <h1 className="landing-title">
          Autonomous Neural <br />
          <span className="gradient-text">Voice Clone Detection</span>
        </h1>

        <p className="landing-subtitle">
          Continuous acoustic verification and deepfake synthesis detection for live
          calls, streams, and virtual conferences using a fine-tuned self-supervised
          speech encoder.
        </p>

        {/* The globe IS the overview: full-bleed, no window chrome around it. */}
        <div className="landing-globe">
          <GlobeStudy mode="dark" scale={1} opacity={1} />
        </div>

        {/* Minimal Action Controls */}
        <div className="landing-cta-group">
          <button
            type="button"
            className="btn btn-primary landing-btn-main"
            onClick={onLaunchDetector}
          >
            <Activity size={18} />
            <span>Launch Live Detector</span>
            <ArrowRight size={16} />
          </button>

          <button
            type="button"
            className="btn btn-secondary landing-btn-sub"
            onClick={onOpenAbout}
          >
            <BookOpen size={16} />
            <span>Architecture &amp; Docs</span>
          </button>

          <button
            type="button"
            className="btn btn-secondary landing-btn-sub"
            onClick={onOpenSettings}
          >
            <Settings2 size={16} />
            <span>Settings</span>
          </button>
        </div>

        {/* Minimal High-Tech Telemetry Bar */}
        <div className="landing-metrics">
          <div className="metric-item">
            <span className="metric-val font-mono">90.9%</span>
            <span className="metric-lbl">Unseen TTS Detection</span>
          </div>
          <div className="metric-sep" />
          <div className="metric-item">
            <span className="metric-val font-mono">&lt; 4s</span>
            <span className="metric-lbl">Sliding Analysis</span>
          </div>
          <div className="metric-sep" />
          <div className="metric-item">
            <span className="metric-val font-mono">16 kHz</span>
            <span className="metric-lbl">PCM Streaming</span>
          </div>
        </div>
      </section>

      {/* Minimal Feature Cards Grid (Monochrome & Zero Green) */}
      <section className="landing-grid-section">
        <div className="section-head">
          <h2>Core Detection Architecture</h2>
          <p>Multi-layered acoustic and biometric validation fused in real time</p>
        </div>

        <div className="landing-cards-grid">
          <div className="card feature-card">
            <div className="feature-icon-box">
              <Cpu size={18} color="#e2e8f0" />
            </div>
            <h3>Fine-Tuned wav2vec2 Encoder</h3>
            <p>
              A self-supervised speech encoder fine-tuned on generator-disjoint spoof corpora, so it
              flags vocoder artefacts from synthesis engines it was never trained on.
            </p>
          </div>

          <div className="card feature-card">
            <div className="feature-icon-box">
              <Radio size={18} color="#e2e8f0" />
            </div>
            <h3>Speaker Biometric Verification</h3>
            <p>
              Matches incoming voice embeddings against enrolled reference profiles using an
              ECAPA-TDNN speaker encoder to prevent human impersonators.
            </p>
          </div>

          <div className="card feature-card">
            <div className="feature-icon-box">
              <Layers size={18} color="#e2e8f0" />
            </div>
            <h3>Instant Decision Fusion</h3>
            <p>
              Combines deep spoof scores, speaker verification, and prosodic metrics with calibrated
              dynamic weights into a unified 0–100 risk score.
            </p>
          </div>

          <div className="card feature-card">
            <div className="feature-icon-box">
              <Globe2 size={18} color="#e2e8f0" />
            </div>
            <h3>Zero-Retention Privacy</h3>
            <p>
              Audio is streamed directly over ephemeral WebSockets and processed in RAM. No voice
              recordings or conversation audio are ever stored on disk.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingView;
