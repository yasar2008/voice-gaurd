'use client';

import React, { useState } from 'react';
import { Home as HomeIcon, MonitorSpeaker, Settings2, Info, ShieldCheck } from 'lucide-react';
import { useDetector } from '@/hooks/useDetector';
import { useStoredFlag } from '@/hooks/useLocalStorage';
import { LandingView } from '@/components/LandingView';
import { LiveDetector } from '@/components/LiveDetector';
import { SettingsView } from '@/components/SettingsView';
import { AboutView } from '@/components/AboutView';
import { ShaderBackground } from '@/components/ui/hero';
import GatewayFlow from '@/components/ui/gateway-flow';
import { Cpu } from 'lucide-react';

type View = 'home' | 'listen' | 'settings' | 'about';

const NAV: { key: View; label: string; Icon: typeof MonitorSpeaker }[] = [
  { key: 'home', label: 'Overview', Icon: HomeIcon },
  { key: 'listen', label: 'Detector', Icon: MonitorSpeaker },
  { key: 'settings', label: 'Settings', Icon: Settings2 },
  { key: 'about', label: 'About', Icon: Info },
];

const DETAIL_KEY = 'vg.showDetail';

export default function Home() {
  const detector = useDetector();
  const [view, setView] = useState<View>('home');
  const [showDetail, setShowDetail] = useStoredFlag(DETAIL_KEY);

  const listening = detector.streamState === 'listening';

  return (
    <div className="shell">
      {/*
        Ambient WebGL plasma backdrop (ShaderBackground). Fixed and pointer-events:none
        so it renders smoothly behind Detector, Settings, and About views.
      */}
      {view !== 'home' && (
        <div className="ambient" aria-hidden>
          <ShaderBackground className="ambient-canvas" />
        </div>
      )}

      <header className="topbar">
        <button
          type="button"
          onClick={() => setView('home')}
          className="brand-btn"
          aria-label="Go to Voice Guard home"
        >
          <div className="brand">
            <div className="brand-logo-icon" aria-hidden>
              <ShieldCheck size={16} strokeWidth={2.4} />
            </div>
            <span className="brand-name">
              <span className="brand-name-primary">VOICE</span>
              <span className="brand-name-accent">GUARD</span>
            </span>
          </div>
        </button>

        {/* Desktop Topbar Navigation */}
        <div className="topbar-nav">
          {NAV.map(({ key, label, Icon }) => (
            <button
              key={key}
              type="button"
              onClick={() => setView(key)}
              className={`topbar-nav-item ${view === key ? 'topbar-nav-item--active' : ''}`}
            >
              <Icon size={14} strokeWidth={1.8} />
              <span>{label}</span>
            </button>
          ))}
          {listening && <span className="chip chip--live">live</span>}
        </div>

        {listening && <span className="chip chip--live mobile-only">live</span>}
      </header>

      <main className="content">
        {view === 'home' && (
          <LandingView
            onLaunchDetector={() => setView('listen')}
            onOpenSettings={() => setView('settings')}
            onOpenAbout={() => setView('about')}
          />
        )}
        {view === 'listen' && <LiveDetector detector={detector} showDetail={showDetail} />}
        {view === 'settings' && (
          <SettingsView
            detector={detector}
            showDetail={showDetail}
            onShowDetailChange={setShowDetail}
          />
        )}
        {view === 'about' && <AboutView />}
      </main>

      {/* Floating Bottom Navigation (Mobile & Tablet) */}
      <nav className="navbar" aria-label="Sections">
        {NAV.map(({ key, label, Icon }) => (
          <button
            key={key}
            type="button"
            onClick={() => setView(key)}
            className={`nav-item ${view === key ? 'nav-item--active' : ''}`}
            aria-label={label}
            aria-current={view === key ? 'page' : undefined}
          >
            <Icon size={16} strokeWidth={1.8} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
