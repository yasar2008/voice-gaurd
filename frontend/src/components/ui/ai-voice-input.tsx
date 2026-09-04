"use client";

import React, { useState, useEffect, useRef, type ReactNode } from "react";
import { Mic } from "lucide-react";
import { cn } from "@/lib/utils";

interface AIVoiceInputProps {
  onStart?: () => void;
  onStop?: (duration: number) => void;
  visualizerBars?: number;
  demoMode?: boolean;
  demoInterval?: number;
  className?: string;
  /**
   * Controlled mode. When supplied, the widget reflects this flag instead of
   * owning the on/off state, and the button only reports intent through
   * onStart/onStop. Capture can fail or be cancelled at the browser's share
   * picker, so the parent — not this component — is the source of truth for
   * whether we are actually recording.
   */
  active?: boolean;
  /** Live capture driving the bars. Without it they fall back to random motion. */
  stream?: MediaStream | null;
  /** Swapped at the call site: this app captures system audio, not a microphone. */
  icon?: ReactNode;
  idleLabel?: string;
  activeLabel?: string;
  disabled?: boolean;
}

const formatTime = (seconds: number) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

export function AIVoiceInput({
  onStart,
  onStop,
  visualizerBars = 48,
  demoMode = false,
  demoInterval = 3000,
  className,
  active,
  stream = null,
  icon,
  idleLabel = "Click to speak",
  activeLabel = "Listening...",
  disabled = false,
}: AIVoiceInputProps) {
  const controlled = active !== undefined;

  const [selfSubmitted, setSelfSubmitted] = useState(false);
  const [isDemo, setIsDemo] = useState(demoMode);

  // Derived, not synced. Mirroring `active` into state would mean a setState
  // inside an effect and a frame where the two disagree.
  const submitted = controlled ? Boolean(active) : selfSubmitted;

  const barsRef = useRef<HTMLDivElement>(null);
  const clockRef = useRef<HTMLSpanElement>(null);
  const elapsedRef = useRef(0);

  // Callbacks live in refs so the run/stop effect can depend on `submitted`
  // alone. Depending on `time` (as the original did) re-ran the effect on every
  // tick, firing onStart once per second and calling onStop on mount.
  const onStartRef = useRef(onStart);
  const onStopRef = useRef(onStop);
  useEffect(() => {
    onStartRef.current = onStart;
    onStopRef.current = onStop;
  });

  // The clock is written straight to the DOM. Held in state it would re-render
  // the whole bar array once a second for a single changing string.
  useEffect(() => {
    if (!submitted) {
      elapsedRef.current = 0;
      if (clockRef.current) clockRef.current.textContent = formatTime(0);
      return;
    }

    const id = setInterval(() => {
      elapsedRef.current += 1;
      if (clockRef.current) clockRef.current.textContent = formatTime(elapsedRef.current);
    }, 1000);

    return () => clearInterval(id);
  }, [submitted]);

  // Uncontrolled mode keeps the original's callback contract.
  const firstRun = useRef(true);
  useEffect(() => {
    if (controlled) return;
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    if (submitted) onStartRef.current?.();
    else onStopRef.current?.(elapsedRef.current);
  }, [submitted, controlled]);

  useEffect(() => {
    if (!isDemo) return;

    let timeoutId: ReturnType<typeof setTimeout>;
    const runAnimation = () => {
      setSelfSubmitted(true);
      timeoutId = setTimeout(() => {
        setSelfSubmitted(false);
        timeoutId = setTimeout(runAnimation, 1000);
      }, demoInterval);
    };

    const initialTimeout = setTimeout(runAnimation, 100);
    return () => {
      clearTimeout(timeoutId);
      clearTimeout(initialTimeout);
    };
  }, [isDemo, demoInterval]);

  // Bar heights are written to the DOM, never rendered. The original computed
  // Math.random() during render, which is impure and produces different heights
  // on any incidental re-render.
  useEffect(() => {
    const host = barsRef.current;
    if (!host) return;

    const bars = Array.from(host.children) as HTMLElement[];

    if (!submitted) {
      bars.forEach((b) => {
        b.style.height = "";
      });
      return;
    }

    // No stream: keep the original's random look, but on a timer.
    if (!stream) {
      const scatter = () => {
        bars.forEach((b) => {
          b.style.height = `${20 + Math.random() * 80}%`;
        });
      };
      scatter();
      const id = setInterval(scatter, 300);
      return () => clearInterval(id);
    }

    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    let ctx: AudioContext;
    try {
      ctx = new AudioCtx();
    } catch {
      return;
    }

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 128;
    analyser.smoothingTimeConstant = 0.75;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(analyser);

    const data = new Uint8Array(analyser.frequencyBinCount);
    let frame = 0;

    const loop = () => {
      frame = requestAnimationFrame(loop);
      analyser.getByteFrequencyData(data);
      for (let i = 0; i < bars.length; i++) {
        const bin = Math.floor((i / bars.length) * data.length);
        bars[i].style.height = `${12 + (data[bin] / 255) * 88}%`;
      }
    };
    loop();

    return () => {
      cancelAnimationFrame(frame);
      source.disconnect();
      if (ctx.state !== "closed") ctx.close();
    };
  }, [submitted, stream]);

  const handleClick = () => {
    if (disabled) return;
    if (isDemo) {
      setIsDemo(false);
      setSelfSubmitted(false);
      return;
    }
    if (controlled) {
      if (active) onStop?.(elapsedRef.current);
      else onStart?.();
      return;
    }
    setSelfSubmitted((prev) => !prev);
  };

  return (
    <div className={cn("ai-voice", submitted && "ai-voice--active", className)}>
      <div className="ai-voice-inner">
        <button
          className={cn("ai-voice-btn", submitted && "ai-voice-btn--active")}
          type="button"
          onClick={handleClick}
          disabled={disabled}
          aria-pressed={submitted}
          aria-label={submitted ? "Stop listening" : "Start listening"}
        >
          {submitted ? (
            <span className="ai-voice-stop" />
          ) : (
            icon ?? <Mic size={24} strokeWidth={1.6} />
          )}
        </button>

        <span
          ref={clockRef}
          className={cn("ai-voice-time", submitted && "ai-voice-time--active")}
        >
          {formatTime(0)}
        </span>

        <div
          className="ai-voice-bars"
          ref={barsRef}
          data-live={stream ? "true" : undefined}
          aria-hidden
        >
          {[...Array(visualizerBars)].map((_, i) => (
            <div
              key={i}
              className={cn("ai-voice-bar", submitted && "ai-voice-bar--active")}
            />
          ))}
        </div>

        <p className="ai-voice-hint">{submitted ? activeLabel : idleLabel}</p>
      </div>
    </div>
  );
}

export default AIVoiceInput;
