'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Mic,
  Square,
  Upload,
  Trash2,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { Detector, DEFAULT_API_BASE } from '@/hooks/useDetector';
import { startWavRecording, WavRecorder } from '@/lib/recorder';
import { RiskMeter } from '@/components/RiskMeter';
import { SignalBreakdown } from '@/components/SignalBreakdown';
import { RiskAnalysis } from '@/types/detector';

// Mirrors backend/config.py FusionConfig — keep in step with it.
const DEFAULT_WEIGHTS = { spoof: 70, speaker: 20, prosody: 10 };
const DEFAULT_THRESHOLD = 65;
const ENROLL_SECONDS = 8;
const CAPTURE_SECONDS = 12;

// ---------------------------------------------------------------- primitives

const Section: React.FC<{
  title: string;
  hint?: string;
  className?: string;
  children: React.ReactNode;
}> = ({ title, hint, className = '', children }) => (
  <section className={`card settings-section ${className}`}>
    <header className="settings-head">
      <h2>{title}</h2>
      {hint && <p>{hint}</p>}
    </header>
    {children}
  </section>
);

const SliderRow: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  suffix?: string;
  onChange: (v: number) => void;
}> = ({ label, value, min, max, suffix = '', onChange }) => (
  <div className="slider-row">
    <div className="slider-head">
      <span>{label}</span>
      <span className="font-mono">
        {value}
        {suffix}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  </div>
);

const Toggle: React.FC<{
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}> = ({ label, hint, checked, onChange }) => (
  <button
    type="button"
    className="toggle-row"
    onClick={() => onChange(!checked)}
    aria-pressed={checked}
  >
    <span>
      <span className="toggle-label">{label}</span>
      {hint && <span className="toggle-hint">{hint}</span>}
    </span>
    <span className={`switch ${checked ? 'switch--on' : ''}`} aria-hidden>
      <span />
    </span>
  </button>
);

const Status: React.FC<{ text: string; error?: boolean }> = ({ text, error }) => (
  <div className={`notice ${error ? 'notice--error' : 'notice--ok'}`}>
    {error ? <AlertCircle size={14} /> : <CheckCircle2 size={14} />}
    <span>{text}</span>
  </div>
);

// ------------------------------------------------------------------- screen

interface SettingsViewProps {
  detector: Detector;
  showDetail: boolean;
  onShowDetailChange: (v: boolean) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  detector,
  showDetail,
  onShowDetailChange,
}) => {
  const {
    apiBase,
    setApiBase,
    config,
    health,
    online,
    calibrated,
    enrolled,
    updateConfig,
    analyzeFile,
    captureSample,
    enroll,
    clearEnrollment,
    streamState,
  } = detector;

  // Drafts stay null until the user touches a control; until then the
  // rendered value is whatever the backend reports.
  const [thresholdDraft, setThresholdDraft] = useState<number | null>(null);
  const [weightsDraft, setWeightsDraft] = useState<typeof DEFAULT_WEIGHTS | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const commitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [enrollState, setEnrollState] = useState<'idle' | 'recording' | 'saving'>('idle');
  const [enrollSeconds, setEnrollSeconds] = useState(0);
  const [enrollMessage, setEnrollMessage] = useState<{ text: string; error?: boolean } | null>(
    null,
  );
  const recorderRef = useRef<WavRecorder | null>(null);
  const enrollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const enrollFileRef = useRef<HTMLInputElement>(null);

  const [testResult, setTestResult] = useState<RiskAnalysis | null>(null);
  const [testing, setTesting] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const testFileRef = useRef<HTMLInputElement>(null);

  const [apiDraft, setApiDraft] = useState<string | null>(null);

  const [capState, setCapState] = useState<'idle' | 'recording' | 'saving'>('idle');
  const [capSeconds, setCapSeconds] = useState(0);
  const [capError, setCapError] = useState<string | null>(null);
  const [capResults, setCapResults] = useState<
    { name: string; risk: number; bonafide: number; rms: number; count: number }[]
  >([]);
  const capRecorderRef = useRef<WavRecorder | null>(null);
  const capTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const threshold =
    thresholdDraft ?? Math.round(config?.alert_threshold ?? DEFAULT_THRESHOLD);
  const weights =
    weightsDraft ??
    (config
      ? {
          spoof: Math.round(config.fusion_weights.spoof_detection * 100),
          speaker: Math.round(config.fusion_weights.speaker_similarity * 100),
          prosody: Math.round(config.fusion_weights.prosody_naturalness * 100),
        }
      : DEFAULT_WEIGHTS);
  const apiValue = apiDraft ?? apiBase;

  useEffect(
    () => () => {
      if (commitTimer.current) clearTimeout(commitTimer.current);
      if (enrollTimerRef.current) clearInterval(enrollTimerRef.current);
      if (capTimerRef.current) clearInterval(capTimerRef.current);
      recorderRef.current?.cancel();
      capRecorderRef.current?.cancel();
    },
    [],
  );

  /** Debounce PUT /config so dragging a slider doesn't spam the backend. */
  const commit = (nextThreshold: number, nextWeights: typeof DEFAULT_WEIGHTS) => {
    if (commitTimer.current) clearTimeout(commitTimer.current);
    commitTimer.current = setTimeout(async () => {
      const total = nextWeights.spoof + nextWeights.speaker + nextWeights.prosody;
      if (total === 0) return;
      const spoof = Number((nextWeights.spoof / total).toFixed(2));
      const speaker = Number((nextWeights.speaker / total).toFixed(2));
      const prosody = Number((1 - spoof - speaker).toFixed(2));
      try {
        setConfigError(null);
        await updateConfig({
          alert_threshold: nextThreshold,
          fusion_weights: {
            spoof_detection: spoof,
            speaker_similarity: speaker,
            prosody_naturalness: prosody,
          },
        });
      } catch (e) {
        setConfigError((e as Error).message);
      }
    }, 300);
  };

  const changeThreshold = (v: number) => {
    setThresholdDraft(v);
    commit(v, weights);
  };

  const changeWeight = (key: keyof typeof DEFAULT_WEIGHTS, v: number) => {
    const next = { ...weights, [key]: v };
    setWeightsDraft(next);
    commit(threshold, next);
  };

  const resetWeights = () => {
    setWeightsDraft(DEFAULT_WEIGHTS);
    setThresholdDraft(DEFAULT_THRESHOLD);
    commit(DEFAULT_THRESHOLD, DEFAULT_WEIGHTS);
  };

  const weightTotal = weights.spoof + weights.speaker + weights.prosody;
  const asPercent = (v: number) => (weightTotal ? Math.round((v / weightTotal) * 100) : 0);

  // --------------------------------------------------------- enrollment
  const stopEnrollRecording = async () => {
    if (enrollTimerRef.current) {
      clearInterval(enrollTimerRef.current);
      enrollTimerRef.current = null;
    }
    const recorder = recorderRef.current;
    recorderRef.current = null;
    if (!recorder) return;

    setEnrollState('saving');
    try {
      const blob = await recorder.stop();
      await enroll(blob);
      setEnrollMessage({ text: 'Reference voice enrolled — 192-d voiceprint stored.' });
    } catch (e) {
      setEnrollMessage({ text: (e as Error).message, error: true });
    } finally {
      setEnrollState('idle');
      setEnrollSeconds(0);
    }
  };

  const startEnrollRecording = async () => {
    setEnrollMessage(null);
    try {
      recorderRef.current = await startWavRecording();
    } catch {
      setEnrollMessage({ text: 'Microphone access denied.', error: true });
      return;
    }
    setEnrollState('recording');
    setEnrollSeconds(0);
    enrollTimerRef.current = setInterval(() => {
      setEnrollSeconds((s) => {
        if (s + 1 >= ENROLL_SECONDS) {
          stopEnrollRecording();
          return ENROLL_SECONDS;
        }
        return s + 1;
      });
    }, 1000);
  };

  const enrollFromFile = async (file: File) => {
    setEnrollState('saving');
    setEnrollMessage(null);
    try {
      await enroll(file);
      setEnrollMessage({ text: 'Reference voice enrolled from file.' });
    } catch (e) {
      setEnrollMessage({ text: (e as Error).message, error: true });
    } finally {
      setEnrollState('idle');
    }
  };

  // ------------------------------------------------- calibration capture
  const stopCapture = async () => {
    if (capTimerRef.current) {
      clearInterval(capTimerRef.current);
      capTimerRef.current = null;
    }
    const recorder = capRecorderRef.current;
    capRecorderRef.current = null;
    if (!recorder) return;

    setCapState('saving');
    try {
      const blob = await recorder.stop();
      const r = await captureSample(blob, 'bonafide');
      setCapResults((prev) => [
        {
          name: r.saved_as,
          risk: r.risk_score,
          bonafide: r.bonafide_score,
          rms: r.rms,
          count: r.count_in_folder,
        },
        ...prev,
      ]);
    } catch (e) {
      setCapError((e as Error).message);
    } finally {
      setCapState('idle');
      setCapSeconds(0);
    }
  };

  const startCapture = async () => {
    setCapError(null);
    try {
      capRecorderRef.current = await startWavRecording();
    } catch {
      setCapError('Microphone access denied.');
      return;
    }
    setCapState('recording');
    setCapSeconds(0);
    capTimerRef.current = setInterval(() => {
      setCapSeconds((s) => {
        if (s + 1 >= CAPTURE_SECONDS) {
          stopCapture();
          return CAPTURE_SECONDS;
        }
        return s + 1;
      });
    }, 1000);
  };

  // --------------------------------------------------------------- test
  const runFileTest = async (file: File) => {
    setTesting(true);
    setTestError(null);
    try {
      setTestResult(await analyzeFile(file));
    } catch (e) {
      setTestError((e as Error).message);
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="view">
      <div className="view-header">
        <h1 className="view-title">System Settings</h1>
        <p className="view-subtitle">Adjust fusion thresholds, speaker profiles, and backend connectivity</p>
      </div>

      <div className="view-grid">
        <Section
          title="Detection sensitivity"
          hint="Risk above this score is called out as a synthetic voice. Lower is stricter."
        >
          <div className="notice">
            <AlertCircle size={14} />
            <span>
              This slider has little effect with the current model. Measured over 74 clips, 99% of
              its outputs were fully saturated at 0.000 or 1.000, so risk lands near 0 or near 88 and
              almost never in between — every threshold from 30 to 90 classifies identically. It is
              kept for the fusion maths and for future models with graded scores.
            </span>
          </div>
          <SliderRow
            label="Alert threshold"
            value={threshold}
            min={30}
            max={90}
            suffix=" / 100"
            onChange={changeThreshold}
          />
          <div className="scale-labels">
            <span>Strict · 30</span>
            <span>Balanced · 65</span>
            <span>Lenient · 90</span>
          </div>
        </Section>

        <Section
          title="Signal weights"
          hint="How much each detector contributes to the fused risk score. Values are normalised to 100%."
        >
          <SliderRow
            label={`Synthesis detection — ${asPercent(weights.spoof)}%`}
            value={weights.spoof}
            min={0}
            max={100}
            onChange={(v) => changeWeight('spoof', v)}
          />
          <SliderRow
            label={`Speaker match — ${asPercent(weights.speaker)}%`}
            value={weights.speaker}
            min={0}
            max={100}
            onChange={(v) => changeWeight('speaker', v)}
          />
          <SliderRow
            label={`Acoustic naturalness — ${asPercent(weights.prosody)}%`}
            value={weights.prosody}
            min={0}
            max={100}
            onChange={(v) => changeWeight('prosody', v)}
          />
          <div className="settings-actions">
            <button type="button" className="btn btn-secondary" onClick={resetWeights}>
              <RotateCcw size={14} /> Reset to defaults
            </button>
            {streamState === 'listening' && (
              <span className="hint">Changes apply to the next live update.</span>
            )}
          </div>
          {configError && <Status text={configError} error />}
        </Section>

        <Section
          title="Reference voice"
          hint="Optional. Enrol the voice you expect to hear so the detector can also flag impostors who sound human."
        >
          <div className="settings-actions">
            {enrollState === 'recording' ? (
              <button type="button" className="btn btn-danger" onClick={stopEnrollRecording}>
                <Square size={14} /> Stop · {enrollSeconds}s / {ENROLL_SECONDS}s
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-primary"
                onClick={startEnrollRecording}
                disabled={enrollState === 'saving'}
              >
                {enrollState === 'saving' ? (
                  <Loader2 size={14} className="spin" />
                ) : (
                  <Mic size={14} />
                )}
                Record {ENROLL_SECONDS}s sample
              </button>
            )}

            <input
              ref={enrollFileRef}
              type="file"
              accept=".wav,.flac,.mp3"
              hidden
              onChange={(e) => e.target.files?.[0] && enrollFromFile(e.target.files[0])}
            />
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => enrollFileRef.current?.click()}
              disabled={enrollState !== 'idle'}
            >
              <Upload size={14} /> Upload sample
            </button>

            {enrolled && (
              <button type="button" className="btn btn-ghost danger" onClick={clearEnrollment}>
                <Trash2 size={14} /> Clear voiceprint
              </button>
            )}
          </div>
          <p className="hint">
            Status:{' '}
            <strong>
              {enrolled
                ? 'voiceprint enrolled'
                : 'no voiceprint — signal excluded, its weight goes to the others'}
            </strong>
          </p>
          {enrollMessage && <Status text={enrollMessage.text} error={enrollMessage.error} />}
        </Section>

        <Section
          title="Build a calibration set"
          hint="Record your real voice to measure how often the detector wrongly flags it. Saves 16 kHz WAV into data/eval/bonafide and scores it immediately."
        >
          <div className="settings-actions">
            {capState === 'recording' ? (
              <button type="button" className="btn btn-danger" onClick={stopCapture}>
                <Square size={14} /> Stop · {capSeconds}s / {CAPTURE_SECONDS}s
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-primary"
                onClick={startCapture}
                disabled={capState === 'saving' || !online}
              >
                {capState === 'saving' ? <Loader2 size={14} className="spin" /> : <Mic size={14} />}
                Record {CAPTURE_SECONDS}s of my voice
              </button>
            )}
            <span className="hint">Speak normally, as you would on a call.</span>
          </div>

          {capError && <Status text={capError} error />}

          {capResults.length > 0 && (
            <>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>saved</th>
                      <th>rms</th>
                      <th>AASIST</th>
                      <th>risk</th>
                      <th>verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {capResults.map((r) => (
                      <tr key={r.name}>
                        <td className="font-mono">{r.name}</td>
                        <td className="font-mono">{r.rms.toFixed(4)}</td>
                        <td className="font-mono">{r.bonafide.toFixed(3)}</td>
                        <td className="font-mono">{r.risk.toFixed(1)}</td>
                        <td style={{ color: r.risk >= threshold ? 'var(--rose)' : 'var(--emerald)' }}>
                          {r.risk >= threshold ? 'flagged synthetic' : 'passed as real'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="hint">
                {capResults[0].count} clip{capResults[0].count === 1 ? '' : 's'} in{' '}
                <code>data/eval/bonafide</code>. Every one flagged as synthetic is a false alarm — get
                5–10, then run <code>python scripts/calibrate.py --bonafide data/eval/bonafide --spoof
                data/eval/spoof_tts</code> for a threshold fitted to your voice.
              </p>
            </>
          )}
        </Section>

        <Section title="Test with a file" hint="Run a recorded WAV or FLAC through the same pipeline.">
          <input
            ref={testFileRef}
            type="file"
            accept=".wav,.flac,.mp3"
            hidden
            onChange={(e) => e.target.files?.[0] && runFileTest(e.target.files[0])}
          />
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => testFileRef.current?.click()}
            disabled={testing}
          >
            {testing ? <Loader2 size={14} className="spin" /> : <Upload size={14} />}
            {testing ? 'Analysing…' : 'Choose audio file'}
          </button>
          {testError && <Status text={testError} error />}
          {testResult && (
            <div className="test-result">
              <RiskMeter
                score={testResult.risk_score}
                alert={testResult.alert}
                confidence={testResult.confidence}
                threshold={threshold}
              />
              <SignalBreakdown analysis={testResult} enrolled={enrolled} />
            </div>
          )}
        </Section>

        <Section
          title="Reliable range"
          hint="Measured limits of the spoof model. Outside these, the app withholds a verdict rather than guessing."
        >
          <dl className="kv">
            <div>
              <dt>Signal-to-noise</dt>
              <dd className="font-mono">≥ 30 dB</dd>
            </div>
            <div>
              <dt>Bandwidth</dt>
              <dd className="font-mono">full-band (not telephone)</dd>
            </div>
            <div>
              <dt>Speech length</dt>
              <dd className="font-mono">≥ 2 s</dd>
            </div>
          </dl>
          <p className="hint">
            Below about 25 dB SNR synthetic speech evades detection entirely — at 15 dB a TTS render
            scored 0.997 &ldquo;genuine&rdquo;. On telephone-band audio the reverse happens and real
            voices score 0.000. Both are silent failures, so the detector now declines instead.
          </p>
        </Section>

        <Section title="Display">
          <Toggle
            label="Show signal breakdown on the home screen"
            hint="Off by default to keep the live view to a single verdict."
            checked={showDetail}
            onChange={onShowDetailChange}
          />
        </Section>

        <Section title="Backend" hint="Where the FastAPI detector service is running.">
          <div className="settings-actions">
            <input
              className="text-input font-mono"
              value={apiValue}
              onChange={(e) => setApiDraft(e.target.value)}
              placeholder={DEFAULT_API_BASE}
              spellCheck={false}
            />
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => {
                setApiBase(apiValue);
                setApiDraft(null);
              }}
              disabled={apiValue.trim().replace(/\/$/, '') === apiBase}
            >
              Save
            </button>
          </div>
          <dl className="kv">
            <div>
              <dt>Status</dt>
              <dd style={{ color: online ? 'var(--emerald)' : 'var(--rose)' }}>
                {online ? 'connected' : 'offline'}
              </dd>
            </div>
            <div>
              <dt>Device</dt>
              <dd className="font-mono">{health?.device ?? '—'}</dd>
            </div>
            <div>
              <dt>Spoof detector</dt>
              <dd
                className="font-mono"
                style={{ color: calibrated === false ? 'var(--amber)' : undefined }}
              >
                {health?.spoof_detector
                  ? `${health.spoof_detector.backend}${
                      health.spoof_detector.calibrated ? '' : ' (uncalibrated)'
                    }`
                  : '—'}
              </dd>
            </div>
            <div>
              <dt>Models loaded</dt>
              <dd className="font-mono">
                {health
                  ? Object.values(health.models_loaded).filter(Boolean).length +
                    ' / ' +
                    Object.keys(health.models_loaded).length
                  : '—'}
              </dd>
            </div>
          </dl>
        </Section>
      </div>
    </div>
  );
};

export default SettingsView;
