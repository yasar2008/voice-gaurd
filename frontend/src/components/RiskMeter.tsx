'use client';

import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert } from 'lucide-react';

interface RiskMeterProps {
  score: number;
  alert: boolean;
  confidence: 'low' | 'medium' | 'high';
  threshold: number;
}

export const RiskMeter: React.FC<RiskMeterProps> = ({
  score,
  alert,
  confidence,
  threshold,
}) => {
  const safeScore = Math.max(0, Math.min(100, isNaN(score) ? 0 : score));

  // Determine state
  let label = 'Authentic Voice';
  let badgeStyle = 'badge-success';
  let Icon = ShieldCheck;
  let color = 'var(--emerald)';

  if (safeScore >= threshold || alert) {
    label = 'Voice Clone Detected';
    badgeStyle = 'badge-destructive';
    Icon = ShieldAlert;
    color = 'var(--rose)';
  } else if (safeScore >= 35) {
    label = 'Suspicious Audio';
    badgeStyle = 'badge-warning';
    Icon = AlertTriangle;
    color = 'var(--amber)';
  }

  return (
    <div
      className="card"
      style={{
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Top mini header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          width: '100%',
          marginBottom: '16px',
        }}
      >
        <span style={{ fontSize: '0.8rem', color: 'var(--muted-foreground)', fontWeight: 500 }}>
          Live Risk Assessment
        </span>
        <span className={`badge ${badgeStyle}`}>
          <Icon size={12} />
          {label}
        </span>
      </div>

      {/* Hero Score Display */}
      <div style={{ margin: '12px 0 16px', display: 'flex', alignItems: 'baseline', gap: '4px' }}>
        <span
          className="font-mono"
          style={{
            fontSize: '4.5rem',
            fontWeight: 700,
            lineHeight: 1,
            letterSpacing: '-0.04em',
            color: color,
            transition: 'color 0.3s ease',
          }}
        >
          {safeScore.toFixed(0)}
        </span>
        <span style={{ fontSize: '1.25rem', color: 'var(--muted-foreground)', fontWeight: 500 }}>
          /100
        </span>
      </div>

      {/* Sleek Minimalist Progress Bar */}
      <div
        style={{
          width: '100%',
          height: '6px',
          background: '#27272a',
          borderRadius: '9999px',
          overflow: 'hidden',
          marginBottom: '14px',
        }}
      >
        <div
          style={{
            width: `${safeScore}%`,
            height: '100%',
            background: color,
            borderRadius: '9999px',
            transition: 'width 0.35s ease, background-color 0.35s ease',
          }}
        />
      </div>

      {/* Meta info footer */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          width: '100%',
          fontSize: '0.75rem',
          color: 'var(--muted-foreground)',
        }}
      >
        <span>Confidence: <strong style={{ color: 'var(--foreground)' }}>{confidence}</strong></span>
        <span>Alert Threshold: <strong style={{ color: 'var(--foreground)' }}>{threshold}</strong></span>
      </div>
    </div>
  );
};
