'use client';

import React, { useRef, useState } from 'react';
import {
  AlertCircle,
  Loader2,
  MonitorSpeaker,
  Mic,
  Upload,
  Trash2,
  SlidersHorizontal,
  UserCheck,
  FileAudio,
  CheckCircle2,
  RotateCcw,
  Cpu,
} from 'lucide-react';
import { Detector } from '@/hooks/useDetector';
import { verdictFor } from '@/lib/verdict';
import { SignalBreakdown } from '@/components/SignalBreakdown';
import { RiskMeter } from '@/components/RiskMeter';
import { WaveformVisualizer } from '@/components/WaveformVisualizer';
import { AIVoiceInput } from '@/components/ui/ai-voice-input';

interface LiveDetectorProps {
  detector: Detector;
  showDetail: boolean;
}

export const LiveDetector: React.FC<LiveDetectorProps> = ({ detector, showDetail }) => {
  const {
    analysis,
    noSpeech,
    unreliable,
    sessionResult,
    streamState,
    micStream,
    error,
    online,
    calibrated,
    enrolled,
    threshold,
    config,
    toggle,
    refresh,
    enroll,
    clearEnrollment,
    analyzeFile,
    updateConfig,
  } = detector;

  const [activeTab, setActiveTab] = useState<'detector' | 'enroll' | 'weights' | 'filescan'>('detector');
  const [enrollMsg, setEnrollMsg] = useState<{ text: string; error?: boolean } | null>(null);
  const [fileMsg, setFileMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const enrollFileRef = useRef<HTMLInputElement>(null);
  const scanFileRef = useRef<HTMLInputElement>(null);

  const listening = streamState === 'listening';
  const blocked = listening && unreliable !== null;

  const emptyResult = sessionResult !== null && sessionResult.windows === 0;
  const verdict =
    sessionResult && !emptyResult
      ? verdictFor(sessionResult.medianRisk, sessionResult.alert, threshold)
      : null;

  const headline = blocked
    ? 'Can’t judge this'
    : emptyResult
      ? 'No result'
      : verdict
        ? verdict.label
        : listening
          ? 'Listening'
          : 'Ready';

  const idleHint =
    'Listens to incoming calls / audio playing on this computer — pick the tab or window and tick “share audio”.';
  const subline = blocked
    ? 'These conditions are outside the detector’s reliable range.'
    : listening && !verdict
      ? noSpeech
        ? 'No speech yet — waiting for someone to talk.'
        : 'Analysing — first verdict in about 15 seconds…'
      : listening && verdict
        ? noSpeech
          ? 'Paused — no speech right now. Verdict so far:'
          : verdict.detail
        : emptyResult
        ? 'Nothing was analysed — no speech came through, or every window fell outside the reliable range.'
        : verdict
          ? verdict.detail
          : idleHint;

  // Handle Enrollment File
  const handleEnrollFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setEnrollMsg(null);
    try {
      await enroll(file);
      setEnrollMsg({ text: 'Reference voice enrolled — 192-d voiceprint active.' });
    } catch (err) {
      setEnrollMsg({ text: (err as Error).message, error: true });
    } finally {
      setBusy(false);
      e.target.value = '';
    }
  };

  // Handle File Scan
  const handleScanFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setFileMsg(null);
    try {
      await analyzeFile(file);
      setFileMsg(`Analyzed ${file.name} successfully.`);
    } catch (err) {
      setFileMsg(`Error: ${(err as Error).message}`);
    } finally {
      setBusy(false);
      e.target.value = '';
    }
  };

  return (
    <section className="stage">
      {/* Nexus Gateway Header Badge */}
      <div className="flex flex-col items-center gap-1.5 mb-6 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/80 border border-slate-700/60 text-[0.7rem] font-mono text-slate-300 shadow-inner">
          <Cpu size={13} className="text-cyan-400" />
          <span className="uppercase tracking-widest">Nexus Gateway Defense Matrix</span>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        </div>
      </div>

      {/* Target anchor for Gateway Flow bezier stream lines to converge at the listening orb center */}
      <div data-voice-target="true" className="inline-block">
        <AIVoiceInput
          active={listening}
          stream={micStream}
          visualizerBars={48}
          disabled={streamState === 'connecting'}
          icon={
            streamState === 'connecting' ? (
              <Loader2 size={24} className="spin" />
            ) : (
              <MonitorSpeaker size={24} strokeWidth={1.6} />
            )
          }
          idleLabel="Click to share audio"
          activeLabel="Listening…"
          onStart={toggle}
          onStop={() => toggle()}
        />
      </div>

      {/* Real-time Spectral Waveform Visualizer */}
      <div className="w-full max-w-md my-3 opacity-90">
        <WaveformVisualizer isRecording={listening} audioStream={micStream} height={50} />
      </div>

      <div className="verdict">
        <h2
          className="verdict-title"
          style={{ color: blocked ? 'var(--amber)' : (verdict?.color ?? undefined) }}
        >
          {headline}
          {listening && <span className="dots" aria-hidden />}
        </h2>
        <p className="verdict-sub">{subline}</p>

        <div className="verdict-meta">
          {sessionResult && !emptyResult ? (
            <>
              <span className="chip font-mono">
                risk {Math.round(sessionResult.medianRisk)}/100
              </span>
              <span className="chip font-mono">
                {Math.round(sessionResult.syntheticShare * 100)}% flagged
              </span>
            </>
          ) : (
            <span className="chip">threshold {threshold}</span>
          )}
          {enrolled ? (
            <span className="chip text-emerald-400 border-emerald-500/30">
              <UserCheck size={12} /> speaker enrolled
            </span>
          ) : (
            <span className="chip opacity-60">no enrolled speaker</span>
          )}
          {blocked && <span className="chip">no verdict</span>}
        </div>
      </div>

      {/* Live Risk Meter Gauge Component */}
      {analysis && (
        <div className="w-full max-w-md my-4">
          <RiskMeter
            score={analysis.risk_score}
            alert={analysis.risk_score >= threshold}
            confidence={analysis.confidence ?? 'high'}
            threshold={threshold}
          />
        </div>
      )}

      {/* Glassmorphic Command & Settings Panel */}
      <div className="w-full max-w-lg my-5 p-5 rounded-2xl bg-slate-950/80 backdrop-blur-2xl border border-white/10 shadow-2xl flex flex-col gap-4 text-left transition-all duration-300 hover:border-white/20">
        
        {/* Segmented Control Tabs */}
        <div className="flex items-center justify-between gap-1.5 p-1.5 rounded-full bg-slate-900/90 border border-white/10 backdrop-blur-xl shadow-lg">
          <button
            type="button"
            className={`btn flex-1 py-2 px-3 rounded-full text-xs font-medium transition-all ${
              activeTab === 'enroll' ? 'btn-primary' : 'btn-secondary'
            }`}
            onClick={() => setActiveTab(activeTab === 'enroll' ? 'detector' : 'enroll')}
          >
            <UserCheck size={14} className={activeTab === 'enroll' ? 'text-slate-950' : 'text-cyan-400'} />
            <span>Voiceprint</span>
          </button>

          <button
            type="button"
            className={`btn flex-1 py-2 px-3 rounded-full text-xs font-medium transition-all ${
              activeTab === 'weights' ? 'btn-primary' : 'btn-secondary'
            }`}
            onClick={() => setActiveTab(activeTab === 'weights' ? 'detector' : 'weights')}
          >
            <SlidersHorizontal size={14} className={activeTab === 'weights' ? 'text-slate-950' : 'text-indigo-400'} />
            <span>Weights</span>
          </button>

          <button
            type="button"
            className={`btn flex-1 py-2 px-3 rounded-full text-xs font-medium transition-all ${
              activeTab === 'filescan' ? 'btn-primary' : 'btn-secondary'
            }`}
            onClick={() => setActiveTab(activeTab === 'filescan' ? 'detector' : 'filescan')}
          >
            <FileAudio size={14} className={activeTab === 'filescan' ? 'text-slate-950' : 'text-emerald-400'} />
            <span>Scan File</span>
          </button>
        </div>

        {/* Hidden Inputs */}
        <input
          ref={enrollFileRef}
          type="file"
          accept="audio/*"
          hidden
          className="hidden"
          style={{ display: 'none' }}
          onChange={handleEnrollFile}
        />
        <input
          ref={scanFileRef}
          type="file"
          accept="audio/*"
          hidden
          className="hidden"
          style={{ display: 'none' }}
          onChange={handleScanFile}
        />

        {/* Tab 1: Voiceprint Enrollment */}
        {activeTab === 'enroll' && (
          <div className="space-y-3 pt-2 text-xs text-slate-300 animate-in fade-in duration-200">
            <div className="flex items-center justify-between bg-black/40 p-2.5 rounded-lg border border-white/5">
              <span className="text-slate-400">Speaker Profile Status:</span>
              <span className="font-mono font-medium px-2 py-0.5 rounded bg-slate-900 border border-slate-700/80 text-emerald-400">
                {enrolled ? '192-d Voiceprint Active' : 'Not Enrolled'}
              </span>
            </div>

            <div className="flex gap-2">
              <button
                type="button"
                disabled={busy}
                onClick={() => enrollFileRef.current?.click()}
                className="btn btn-secondary flex-1 py-2.5 px-4 rounded-xl text-xs font-medium flex items-center justify-center gap-2 cursor-pointer"
              >
                {busy ? <Loader2 size={14} className="spin" /> : <Upload size={14} className="text-cyan-400" />}
                <span>Upload Reference Voice WAV</span>
              </button>

              {enrolled && (
                <button
                  type="button"
                  onClick={clearEnrollment}
                  className="btn btn-ghost danger py-2.5 px-3 rounded-xl text-xs font-medium flex items-center gap-1.5 cursor-pointer"
                >
                  <Trash2 size={14} />
                  <span>Clear</span>
                </button>
              )}
            </div>

            {enrollMsg && (
              <div className={`p-2.5 rounded-lg border text-xs flex items-center gap-2 ${enrollMsg.error ? 'bg-rose-950/50 border-rose-600/40 text-rose-300' : 'bg-emerald-950/50 border-emerald-600/40 text-emerald-300'}`}>
                {enrollMsg.error ? <AlertCircle size={14} /> : <CheckCircle2 size={14} />}
                <span>{enrollMsg.text}</span>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Weights & Threshold */}
        {activeTab === 'weights' && (
          <div className="space-y-4 pt-2 text-xs text-slate-300 animate-in fade-in duration-200">
            <div className="space-y-2 bg-black/40 p-3 rounded-xl border border-white/5">
              <div className="flex justify-between font-mono text-slate-200 font-medium">
                <span>Alert Risk Threshold</span>
                <span className="text-cyan-400">{threshold}</span>
              </div>
              <input
                type="range"
                min={20}
                max={90}
                value={threshold}
                onChange={(e) => updateConfig({ alert_threshold: Number(e.target.value) })}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <div className="flex justify-between text-[0.68rem] text-slate-500 font-mono">
                <span>Sensitive (20)</span>
                <span>Strict (90)</span>
              </div>
            </div>

            <div className="space-y-2 bg-black/40 p-3 rounded-xl border border-white/5">
              <div className="flex justify-between font-mono text-slate-200 font-medium">
                <span>Spoof Detection Weight</span>
                <span className="text-indigo-400">{Math.round((config?.fusion_weights.spoof_detection ?? 0.7) * 100)}%</span>
              </div>
              <input
                type="range"
                min={10}
                max={90}
                value={Math.round((config?.fusion_weights.spoof_detection ?? 0.7) * 100)}
                onChange={(e) => {
                  const val = Number(e.target.value) / 100;
                  const rem = (1 - val) / 2;
                  updateConfig({
                    fusion_weights: {
                      spoof_detection: val,
                      speaker_similarity: rem,
                      prosody_naturalness: rem,
                    },
                  });
                }}
                className="w-full accent-indigo-400 cursor-pointer"
              />
            </div>
          </div>
        )}

        {/* Tab 3: File Scan */}
        {activeTab === 'filescan' && (
          <div className="space-y-3 pt-2 text-xs text-slate-300 animate-in fade-in duration-200">
            <button
              type="button"
              disabled={busy}
              onClick={() => scanFileRef.current?.click()}
              className="btn btn-primary w-full py-2.5 px-4 rounded-xl text-xs font-medium flex items-center justify-center gap-2 cursor-pointer"
            >
              {busy ? <Loader2 size={16} className="spin" /> : <Upload size={16} />}
              <span>Select Audio File (WAV / FLAC) for Multi-Signal Scan</span>
            </button>

            {fileMsg && (
              <div className="p-2.5 rounded-lg bg-slate-900 border border-white/10 text-xs text-slate-300 flex items-center gap-2">
                <CheckCircle2 size={14} className="text-emerald-400" />
                <span>{fileMsg}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {calibrated === false && (
        <div className="notice notice--warn">
          <AlertCircle size={14} />
          <span>
            Spoof detector is running on uncalibrated weights, so verdicts are not meaningful. Run{' '}
            <code>python scripts/download_checkpoints.py</code> and restart the backend.
          </span>
        </div>
      )}

      {!listening && !online && (
        <button type="button" onClick={refresh} className="notice notice--warn">
          <AlertCircle size={14} />
          <span>Detector backend is offline. Start it, then tap to retry.</span>
        </button>
      )}

      {error && (
        <div className="notice notice--error">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      {blocked && unreliable && (
        <div className="notice notice--warn">
          <AlertCircle size={14} />
          <span>
            {unreliable.reasons[0]}
            {unreliable.reasons.length > 1 && ` (+${unreliable.reasons.length - 1} more)`}
            <br />
            <span className="font-mono" style={{ opacity: 0.75, fontSize: '0.72rem' }}>
              SNR {unreliable.snr_db} dB · HF {unreliable.high_freq_db} dB
            </span>
          </span>
        </div>
      )}

      {!blocked && analysis && analysis.anomalies?.length > 0 && (
        <ul className="anomalies">
          {analysis.anomalies.slice(0, 3).map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      )}

      {(showDetail || analysis) && analysis && (
        <div className="detail-slot w-full max-w-md mt-4">
          <SignalBreakdown analysis={analysis} enrolled={enrolled} />
        </div>
      )}
    </section>
  );
};

export default LiveDetector;
