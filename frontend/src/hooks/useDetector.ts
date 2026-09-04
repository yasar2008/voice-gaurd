'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { AudioQuality, ConfigData, RiskAnalysis, SystemHealth } from '@/types/detector';
import { useStoredValue } from '@/hooks/useLocalStorage';

export const DEFAULT_API_BASE = 'http://localhost:8000';
const API_BASE_KEY = 'vg.apiBase';

export type StreamState = 'idle' | 'connecting' | 'listening';

/**
 * Verdict for a whole listening session rather than one 4-second window.
 * Single windows disagree with each other constantly, so a live headline
 * flickers between "real" and "synthetic" and means nothing on its own.
 */
export interface SessionResult {
  /** Windows the backend actually scored (silence and refusals excluded). */
  windows: number;
  /** Median is the decision: one freak window cannot move it, a mean can. */
  medianRisk: number;
  meanRisk: number;
  /** Share of windows at or above the alert threshold, for context. */
  syntheticShare: number;
  alert: boolean;
  durationS: number;
}

const wsUrlFor = (apiBase: string) =>
  `${apiBase.replace(/\/$/, '').replace(/^http/, 'ws')}/ws/analyze`;

/**
 * Owns everything the UI needs: backend health/config, the live WebSocket
 * microphone stream, one-off file analysis, and speaker enrollment.
 * Views stay presentational.
 */
/** Minimum listening time before a verdict is shown, in ms. */
const FIRST_VERDICT_MS = 15_000;
/** ...and a floor on scored clips, so a near-silent 20s cannot publish one. */
const MIN_CLIPS = 3;

function summarise(
  scores: number[],
  threshold: number,
  startedAt: number | null,
): SessionResult {
  const sorted = [...scores].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const flagged = scores.filter((v) => v >= threshold).length;
  return {
    windows: scores.length,
    medianRisk: median,
    meanRisk: mean,
    syntheticShare: flagged / scores.length,
    alert: median >= threshold,
    durationS: startedAt ? Math.round((Date.now() - startedAt) / 1000) : 0,
  };
}

export function useDetector() {
  const [apiBase, setStoredApiBase] = useStoredValue(API_BASE_KEY, DEFAULT_API_BASE);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [analysis, setAnalysis] = useState<RiskAnalysis | null>(null);
  const [streamState, setStreamState] = useState<StreamState>('idle');
  const [micStream, setMicStream] = useState<MediaStream | null>(null);
  const [elapsedS, setElapsedS] = useState(0);
  const [error, setError] = useState<string | null>(null);
  // Displayed score is smoothed; the verdict needs agreement across frames.
  const [smoothedScore, setSmoothedScore] = useState<number | null>(null);
  const [highStreak, setHighStreak] = useState(0);
  const [noSpeech, setNoSpeech] = useState(false);
  // Set when the backend refuses to judge the current conditions.
  const [unreliable, setUnreliable] = useState<AudioQuality | null>(null);
  // Session aggregate, published only once listening stops.
  const [sessionResult, setSessionResult] = useState<SessionResult | null>(null);

  const apiBaseRef = useRef(apiBase);
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const thresholdRef = useRef(65);
  const sessionScoresRef = useRef<number[]>([]);
  const startedAtRef = useRef<number | null>(null);

  useEffect(() => {
    apiBaseRef.current = apiBase;
  }, [apiBase]);

  useEffect(() => {
    thresholdRef.current = config?.alert_threshold ?? 65;
  }, [config]);

  const setApiBase = useCallback(
    (value: string) => {
      setStoredApiBase(value.trim().replace(/\/$/, '') || DEFAULT_API_BASE);
    },
    [setStoredApiBase],
  );

  // ---------------------------------------------------------------- health
  const refresh = useCallback(async () => {
    try {
      const [hRes, cRes] = await Promise.all([
        fetch(`${apiBaseRef.current}/health`),
        fetch(`${apiBaseRef.current}/config`),
      ]);
      if (hRes.ok) setHealth(await hRes.json());
      if (cRes.ok) setConfig(await cRes.json());
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, [refresh, apiBase]);

  // ------------------------------------------------------------ live audio
  const stop = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close();
    }
    audioCtxRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setMicStream(null);
    setStreamState('idle');
    setElapsedS(0);
    setSmoothedScore(null);
    setHighStreak(0);
    setNoSpeech(false);
    setUnreliable(null);

    // Publish the session verdict. Median over every scored window, matching
    // how the REST path already reduces its 5 windows, so a single outlier
    // cannot decide the result.
    const scores = sessionScoresRef.current;
    if (scores.length > 0) {
      setSessionResult(summarise(scores, thresholdRef.current, startedAtRef.current));
    } else if (startedAtRef.current !== null) {
      // Ran but scored nothing: silence throughout, or every window refused by
      // the quality gate. Recording an empty result keeps the UI from falling
      // back to its idle copy, which would read as a clean pass.
      setSessionResult({
        windows: 0,
        medianRisk: 0,
        meanRisk: 0,
        syntheticShare: 0,
        alert: false,
        durationS: Math.round((Date.now() - startedAtRef.current) / 1000),
      });
    }
    sessionScoresRef.current = [];
    startedAtRef.current = null;
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setAnalysis(null);
    setSmoothedScore(null);
    setHighStreak(0);
    setNoSpeech(false);
    setUnreliable(null);
    setSessionResult(null);
    sessionScoresRef.current = [];
    startedAtRef.current = Date.now();
    setStreamState('connecting');

    // System/tab audio is the only capture path. A caller's voice — in a Meet
    // call or anywhere else — arrives at the speaker, not the microphone, so a
    // mic would record the wrong side of the conversation plus whatever the room
    // adds. Captured system audio is the caller's signal digitally, with no
    // acoustic path and comfortably inside the detector's measured SNR range.
    //
    // Every browser DSP stage is off deliberately: noise suppression is
    // aggressive spectral gating that strips the low-level texture the detector
    // reads as evidence of a real recording, and AGC rewrites the dynamics the
    // prosody extractor measures.
    let stream: MediaStream;
    try {
      // Chrome only offers system/tab audio through getDisplayMedia, and only
      // alongside a video track — which we stop immediately, since we want the
      // sound, not the picture.
      const display = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      display.getVideoTracks().forEach((t) => {
        t.stop();
        display.removeTrack(t);
      });
      if (display.getAudioTracks().length === 0) {
        display.getTracks().forEach((t) => t.stop());
        setStreamState('idle');
        setError(
          'No audio was shared. Re-pick the tab or window and tick "Share tab audio" / ' +
            '"Share system audio" in the picker.',
        );
        return;
      }
      stream = display;
    } catch {
      setStreamState('idle');
      // Capture never began, so this is not a session: clear the stamp or the
      // next real session would measure its duration from this attempt.
      startedAtRef.current = null;
      setError('Screen or tab sharing was cancelled, so there is no audio to listen to.');
      return;
    }

    // If the user stops sharing from Chrome's own bar, tear down cleanly.
    stream.getAudioTracks().forEach((t) => {
      t.onended = () => stop();
    });

    streamRef.current = stream;
    setMicStream(stream);

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrlFor(apiBaseRef.current));
    } catch {
      stop();
      setError('Could not reach the detector backend.');
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      setStreamState('listening');
      setElapsedS(0);
      tickRef.current = setInterval(() => setElapsedS((s) => s + 1), 1000);

      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // Downconvert float32 [-1,1] to 16-bit signed LE PCM, which is what
      // backend/api/ws.py expects on the wire.
      processor.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm = new DataView(new ArrayBuffer(input.length * 2));
        for (let i = 0; i < input.length; i++) {
          const s = Math.max(-1, Math.min(1, input[i]));
          pcm.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        ws.send(pcm.buffer);
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'risk_update') {
          const update = data as RiskAnalysis;
          setAnalysis(update);
          setNoSpeech(false);
          setUnreliable(null);
          sessionScoresRef.current.push(update.risk_score);
          // Publish a running verdict once there is enough audio behind it.
          // Median over 20s+ barely moves between updates, so this reads as a
          // settled result that firms up -- not the per-window flicker that
          // made the old live headline meaningless.
          const started = startedAtRef.current;
          const scores = sessionScoresRef.current;
          if (
            scores.length >= MIN_CLIPS &&
            started !== null &&
            Date.now() - started >= FIRST_VERDICT_MS
          ) {
            setSessionResult(summarise(scores, thresholdRef.current, started));
          }
          // EMA so one bad 4-second window cannot swing the number on screen.
          setSmoothedScore((prev) =>
            prev === null ? update.risk_score : prev * 0.5 + update.risk_score * 0.5,
          );
          // Hysteresis: calling it synthetic needs two consecutive frames to agree.
          setHighStreak((n) => (update.risk_score >= thresholdRef.current ? n + 1 : 0));
        }
        if (data.type === 'idle') setNoSpeech(true);
        if (data.type === 'unreliable') {
          setNoSpeech(false);
          setUnreliable(data.quality as AudioQuality);
        }
        if (data.type === 'error') setError(data.message ?? 'Backend error');
      } catch {
        // ignore malformed frame
      }
    };

    ws.onerror = () => setError('Lost connection to the detector backend.');
    ws.onclose = () => stop();
  }, [stop]);

  const toggle = useCallback(() => {
    if (streamState === 'idle') start();
    else stop();
  }, [streamState, start, stop]);

  useEffect(() => stop, [stop]);

  // ------------------------------------------------------------ file / API
  const analyzeFile = useCallback(async (file: File | Blob) => {
    const body = new FormData();
    body.append('file', file, 'sample.wav');
    const res = await fetch(`${apiBaseRef.current}/analyze`, { method: 'POST', body });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail || 'Analysis failed');
    }
    const data: RiskAnalysis = await res.json();
    setAnalysis(data);
    return data;
  }, []);

  const enroll = useCallback(
    async (file: File | Blob) => {
      const body = new FormData();
      body.append('file', file, 'enrollment.wav');
      const res = await fetch(`${apiBaseRef.current}/enroll`, { method: 'POST', body });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || 'Enrollment failed');
      }
      await refresh();
      return res.json();
    },
    [refresh],
  );

  /** Save a recording into data/eval/<label> and get its score back. */
  const captureSample = useCallback(async (file: File | Blob, label: 'bonafide' | 'spoof') => {
    const body = new FormData();
    body.append('file', file, 'capture.wav');
    body.append('label', label);
    const res = await fetch(`${apiBaseRef.current}/capture`, { method: 'POST', body });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail || 'Could not save the clip');
    }
    return res.json();
  }, []);

  const clearEnrollment = useCallback(async () => {
    await fetch(`${apiBaseRef.current}/enroll`, { method: 'DELETE' });
    await refresh();
  }, [refresh]);

  const updateConfig = useCallback(
    async (patch: {
      alert_threshold?: number;
      fusion_weights?: ConfigData['fusion_weights'];
    }) => {
      const res = await fetch(`${apiBaseRef.current}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || 'Could not update configuration');
      }
      const data = await res.json();
      setConfig((prev) =>
        prev
          ? { ...prev, alert_threshold: data.alert_threshold, fusion_weights: data.fusion_weights }
          : prev,
      );
      return data;
    },
    [],
  );

  return {
    apiBase,
    setApiBase,
    health,
    config,
    analysis,
    smoothedScore,
    /** True only once consecutive windows agree the audio is synthetic. */
    stableAlert: highStreak >= 2,
    noSpeech,
    unreliable,
    /** Whole-session verdict, set when listening stops. Null while running. */
    sessionResult,
    streamState,
    micStream,
    elapsedS,
    error,
    setError,
    online: health?.status === 'ok',
    // null until /health answers; false only when the backend says so.
    calibrated: health ? (health.spoof_detector?.calibrated ?? true) : null,
    enrolled: health?.enrolled_speaker ?? false,
    threshold: config?.alert_threshold ?? 65,
    start,
    stop,
    toggle,
    refresh,
    analyzeFile,
    captureSample,
    enroll,
    clearEnrollment,
    updateConfig,
  };
}

export type Detector = ReturnType<typeof useDetector>;
