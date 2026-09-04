'use client';

import React from 'react';
import { RiskAnalysis } from '@/types/detector';
import { Cpu, UserCheck, Activity } from 'lucide-react';

interface SignalBreakdownProps {
  analysis: RiskAnalysis | null;
  enrolled: boolean;
}

export const SignalBreakdown: React.FC<SignalBreakdownProps> = ({ analysis, enrolled }) => {
  const breakdown = analysis?.breakdown;

  const spoof = breakdown?.spoof_detection;
  const speaker = breakdown?.speaker_similarity;
  const prosodySignal = breakdown?.prosody_naturalness;

  const spoofContrib = spoof?.contribution ?? 0;
  const speakerContrib = speaker?.contribution ?? 0;
  const prosodyContrib = prosodySignal?.contribution ?? 0;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
      {/* Signal 1: AASIST-L */}
      <div className="card card-hover" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="var(--muted-foreground)" />
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Synthesis Model</span>
          </div>
          <span className="font-mono" style={{ fontSize: '0.8rem', color: spoofContrib > 15 ? 'var(--rose)' : 'var(--emerald)' }}>
            +{spoofContrib.toFixed(1)} pts
          </span>
        </div>
        <div style={{ width: '100%', height: '4px', background: '#27272a', borderRadius: '2px', overflow: 'hidden', margin: '8px 0' }}>
          <div
            style={{
              width: `${Math.min(100, spoofContrib * 2)}%`,
              height: '100%',
              background: spoofContrib > 15 ? 'var(--rose)' : 'var(--emerald)',
              transition: 'width 0.3s ease',
            }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>
          <span>AASIST-L (85k params)</span>
          <span>{spoof ? `${(spoof.raw_score * 100).toFixed(0)}% genuine` : 'Waiting...'}</span>
        </div>
      </div>

      {/* Signal 2: ECAPA-TDNN Speaker Verification */}
      <div className="card card-hover" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <UserCheck size={16} color="var(--muted-foreground)" />
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Speaker Match</span>
          </div>
          <span className="font-mono" style={{ fontSize: '0.8rem', color: speakerContrib > 10 ? 'var(--rose)' : 'var(--emerald)' }}>
            +{speakerContrib.toFixed(1)} pts
          </span>
        </div>
        <div style={{ width: '100%', height: '4px', background: '#27272a', borderRadius: '2px', overflow: 'hidden', margin: '8px 0' }}>
          <div
            style={{
              width: `${Math.min(100, speakerContrib * 3.3)}%`,
              height: '100%',
              background: speakerContrib > 10 ? 'var(--rose)' : 'var(--emerald)',
              transition: 'width 0.3s ease',
            }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>
          <span>{enrolled ? 'Enrolled Profile' : 'No Enrolled Voice'}</span>
          <span>{speaker ? `${(speaker.raw_score * 100).toFixed(0)}% match` : 'Neutral (50%)'}</span>
        </div>
      </div>

      {/* Signal 3: Prosody Naturalness */}
      <div className="card card-hover" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} color="var(--muted-foreground)" />
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Acoustic Naturalness</span>
          </div>
          <span className="font-mono" style={{ fontSize: '0.8rem', color: prosodyContrib > 8 ? 'var(--rose)' : 'var(--emerald)' }}>
            +{prosodyContrib.toFixed(1)} pts
          </span>
        </div>
        <div style={{ width: '100%', height: '4px', background: '#27272a', borderRadius: '2px', overflow: 'hidden', margin: '8px 0' }}>
          <div
            style={{
              width: `${Math.min(100, prosodyContrib * 5)}%`,
              height: '100%',
              background: prosodyContrib > 8 ? 'var(--rose)' : 'var(--emerald)',
              transition: 'width 0.3s ease',
            }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>
          <span>Praat F0 & Jitter</span>
          <span>{prosodySignal ? `${(prosodySignal.raw_score * 100).toFixed(0)}% natural` : 'Waiting...'}</span>
        </div>
      </div>
    </div>
  );
};
