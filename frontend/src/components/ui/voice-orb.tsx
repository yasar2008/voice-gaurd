"use client";

import React, { useEffect, useRef, useState } from "react";

export interface VoiceOrbProps {
  /** Optional active audio stream to react to */
  stream?: MediaStream | null;
  /** State mode for orb animation */
  state?: "idle" | "listening" | "speaking" | "alert" | "uncertain";
  /** Base size in pixels (default: 260) */
  size?: number;
  /** Primary glow color or accent */
  color?: string;
  /** Custom class names */
  className?: string;
  /** Interactive click callback */
  onClick?: () => void;
  /** Display label underneath */
  label?: string;
}

export function VoiceOrb({
  stream = null,
  state = "idle",
  size = 240,
  color,
  className = "",
  onClick,
  label,
}: VoiceOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);

  // Audio Analyser Setup
  useEffect(() => {
    if (!stream) {
      if (sourceRef.current) {
        sourceRef.current.disconnect();
        sourceRef.current = null;
      }
      return;
    }

    try {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = audioCtxRef.current || new AudioCtx();
      audioCtxRef.current = ctx;

      if (ctx.state === "suspended") {
        ctx.resume();
      }

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      const source = ctx.createMediaStreamSource(stream);
      source.connect(analyser);
      sourceRef.current = source;
    } catch {
      // Ignore audio context errors if permissions pending
    }

    return () => {
      if (sourceRef.current) {
        sourceRef.current.disconnect();
      }
    };
  }, [stream]);

  // Canvas Orb Rendering Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId = 0;
    let time = 0;
    let smoothedVolume = 0;

    const dataArray = new Uint8Array(128);

    const render = () => {
      animId = requestAnimationFrame(render);
      time += 0.025;

      // Extract Audio Volume
      let volume = 0;
      if (analyserRef.current) {
        analyserRef.current.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < 32; i++) {
          sum += dataArray[i];
        }
        volume = Math.min(1.5, (sum / 32 / 255) * 2.2);
      } else if (state === "speaking" || state === "listening") {
        // Simulated natural breathing pulsation if no microphone connected
        volume = 0.15 + 0.12 * Math.sin(time * 2.5);
      } else {
        volume = 0.04 + 0.03 * Math.sin(time * 1.2);
      }

      smoothedVolume = smoothedVolume * 0.85 + volume * 0.15;
      setAudioLevel(smoothedVolume);

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = size * dpr;
      const h = size * dpr;

      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      ctx.clearRect(0, 0, w, h);
      ctx.save();
      ctx.scale(dpr, dpr);

      const cx = size / 2;
      const cy = size / 2;
      const baseRadius = size * 0.28;
      const dynamicRadius = baseRadius + smoothedVolume * (size * 0.12);

      // Color Theme Palette
      let primaryColor = "16, 185, 129"; // Emerald default
      let secondaryColor = "6, 182, 212"; // Cyan
      let outerGlowColor = "52, 211, 153";

      if (state === "alert") {
        primaryColor = "244, 63, 94"; // Rose/Red
        secondaryColor = "239, 68, 68";
        outerGlowColor = "251, 113, 133";
      } else if (state === "uncertain") {
        primaryColor = "245, 158, 11"; // Amber
        secondaryColor = "251, 191, 36";
        outerGlowColor = "252, 211, 77";
      } else if (state === "listening") {
        primaryColor = "16, 185, 129";
        secondaryColor = "14, 165, 233";
        outerGlowColor = "56, 189, 248";
      } else if (color) {
        // Custom color
        primaryColor = color;
      }

      // 1. Ambient Outer Halo
      const outerGlow = ctx.createRadialGradient(
        cx,
        cy,
        baseRadius * 0.5,
        cx,
        cy,
        dynamicRadius * 1.8,
      );
      outerGlow.addColorStop(0, `rgba(${primaryColor}, ${0.35 + smoothedVolume * 0.4})`);
      outerGlow.addColorStop(0.5, `rgba(${secondaryColor}, ${0.15 + smoothedVolume * 0.2})`);
      outerGlow.addColorStop(1, "rgba(0, 0, 0, 0)");

      ctx.fillStyle = outerGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, dynamicRadius * 1.8, 0, Math.PI * 2);
      ctx.fill();

      // 2. Harmonic Fluid Waves (3 Layers)
      const layers = 3;
      for (let l = 0; l < layers; l++) {
        ctx.beginPath();
        const layerRadius = dynamicRadius * (0.85 + l * 0.12);
        const waveCount = 8 + l * 2;
        const waveAmp = (6 + l * 4) * (0.5 + smoothedVolume * 1.8);
        const speed = time * (1.2 + l * 0.4) * (l % 2 === 0 ? 1 : -1);

        for (let i = 0; i <= 360; i += 4) {
          const rad = (i * Math.PI) / 180;
          const offset =
            Math.sin(rad * waveCount + speed) * waveAmp +
            Math.cos(rad * (waveCount * 0.5) - speed * 0.8) * (waveAmp * 0.5);
          const r = layerRadius + offset;
          const x = cx + r * Math.cos(rad);
          const y = cy + r * Math.sin(rad);

          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();

        const strokeGrad = ctx.createLinearGradient(0, cy - dynamicRadius, 0, cy + dynamicRadius);
        strokeGrad.addColorStop(
          0,
          `rgba(${primaryColor}, ${0.5 - l * 0.12 + smoothedVolume * 0.3})`,
        );
        strokeGrad.addColorStop(
          1,
          `rgba(${secondaryColor}, ${0.2 - l * 0.05 + smoothedVolume * 0.2})`,
        );
        ctx.strokeStyle = strokeGrad;
        ctx.lineWidth = 1.5 + l * 0.6;
        ctx.stroke();
      }

      // 3. Central Core Sphere with Glass Specular Highlights
      const coreGrad = ctx.createRadialGradient(
        cx - dynamicRadius * 0.28,
        cy - dynamicRadius * 0.32,
        dynamicRadius * 0.08,
        cx,
        cy,
        dynamicRadius * 0.95,
      );
      coreGrad.addColorStop(0, `rgba(255, 255, 255, ${0.85 + smoothedVolume * 0.15})`);
      coreGrad.addColorStop(0.25, `rgba(${outerGlowColor}, 0.9)`);
      coreGrad.addColorStop(0.65, `rgba(${primaryColor}, 0.85)`);
      coreGrad.addColorStop(1, `rgba(${secondaryColor}, 0.2)`);

      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, dynamicRadius * 0.78, 0, Math.PI * 2);
      ctx.fill();

      // 4. Subtle Inner Core Ring
      ctx.beginPath();
      ctx.arc(cx, cy, dynamicRadius * 0.82, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.restore();
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [size, state, color]);

  return (
    <div
      className={`voice-orb-container ${className}`}
      onClick={onClick}
      style={{
        width: size,
        height: size,
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        cursor: onClick ? "pointer" : "default",
      }}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: size,
          height: size,
          display: "block",
          filter: `drop-shadow(0 0 ${16 + audioLevel * 20}px rgba(16, 185, 129, ${
            0.3 + audioLevel * 0.4
          }))`,
          transition: "filter 0.2s ease",
        }}
      />
      {label && (
        <span
          className="voice-orb-label"
          style={{
            position: "absolute",
            bottom: -18,
            fontSize: "0.78rem",
            fontWeight: 600,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: "var(--muted-foreground)",
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}

export default VoiceOrb;
