'use client';

import React, { useEffect, useRef } from 'react';

interface WaveformVisualizerProps {
  isRecording: boolean;
  audioStream: MediaStream | null;
  height?: number;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  isRecording,
  audioStream,
  height = 70,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    if (isRecording && audioStream) {
      try {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const audioCtx = new AudioCtx();
        audioCtxRef.current = audioCtx;

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 128;
        analyser.smoothingTimeConstant = 0.85;
        analyserRef.current = analyser;

        const source = audioCtx.createMediaStreamSource(audioStream);
        source.connect(analyser);
        sourceRef.current = source;

        const bufferLength = analyser.frequencyBinCount;
        const freqData = new Uint8Array(bufferLength);

        const draw = () => {
          animFrameRef.current = requestAnimationFrame(draw);
          analyser.getByteFrequencyData(freqData);

          ctx.clearRect(0, 0, rect.width, height);

          // Clean minimalist vertical frequency bars
          const numBars = 48;
          const barWidth = 3;
          const gap = (rect.width - numBars * barWidth) / (numBars - 1);

          for (let i = 0; i < numBars; i++) {
            const dataIndex = Math.floor((i / numBars) * bufferLength);
            const val = freqData[dataIndex] / 255;
            const barHeight = Math.max(3, val * (height - 8));
            const x = i * (barWidth + gap);
            const y = (height - barHeight) / 2;

            ctx.fillStyle = val > 0.6 ? '#fafafa' : val > 0.2 ? '#a1a1aa' : '#3f3f46';
            ctx.beginPath();
            ctx.roundRect(x, y, barWidth, barHeight, 2);
            ctx.fill();
          }
        };

        draw();
      } catch (err) {
        console.error('AudioContext error:', err);
      }
    } else {
      // Clean idle subtle wave animation
      let phase = 0;
      const drawIdle = () => {
        animFrameRef.current = requestAnimationFrame(drawIdle);
        phase += 0.03;

        ctx.clearRect(0, 0, rect.width, height);

        const numBars = 48;
        const barWidth = 3;
        const gap = (rect.width - numBars * barWidth) / (numBars - 1);

        for (let i = 0; i < numBars; i++) {
          const x = i * (barWidth + gap);
          const h = 4 + Math.sin(i * 0.2 + phase) * 3;
          const y = (height - h) / 2;

          ctx.fillStyle = '#27272a';
          ctx.beginPath();
          ctx.roundRect(x, y, barWidth, h, 2);
          ctx.fill();
        }
      };
      drawIdle();
    }

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (sourceRef.current) sourceRef.current.disconnect();
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        audioCtxRef.current.close();
      }
    };
  }, [isRecording, audioStream, height]);

  return (
    <div style={{ width: '100%', height: `${height}px`, position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  );
};
