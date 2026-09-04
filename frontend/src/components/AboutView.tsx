'use client';

import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert, ArrowDown } from 'lucide-react';
import GlobeStudy from '@/components/ui/globe-study';

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <section className="card settings-section">
    <header className="settings-head">
      <h2>{title}</h2>
    </header>
    {children}
  </section>
);

const FlowNode: React.FC<{ title: string; sub?: string }> = ({ title, sub }) => (
  <div className="flow-node">
    <span className="flow-title">{title}</span>
    {sub && <span className="flow-sub">{sub}</span>}
  </div>
);

const Arrow = () => (
  <div className="flow-arrow" aria-hidden>
    <ArrowDown size={14} />
  </div>
);

export const AboutView: React.FC = () => (
  <div className="view">
    <div className="view-header">
      <h1 className="view-title">About Voice Guard</h1>
      <p className="view-subtitle">Architecture, detection telemetry, and benchmark evaluation</p>
    </div>

    <div className="view-grid">
      <Section title="What this does">
      <p className="prose">
        Voice Guard listens to speech in real time and tells you whether the voice is a real human
        or a synthetic clone — the kind produced by text-to-speech and voice-conversion tools used
        in impersonation scams. It is a single verdict on the home screen; everything adjustable
        lives in Settings.
      </p>
    </Section>

    <Section title="How to use it">
      <ol className="steps">
        <li>
          <strong>Fetch the pretrained detector once.</strong> Run{' '}
          <code>python scripts/download_checkpoints.py</code>. Without{' '}
          <code>checkpoints/finetuned_encoder</code> the spoof model falls back to random weights
          and the app says so instead of pretending to have a verdict.
        </li>
        <li>
          <strong>Start the detector backend.</strong> Run{' '}
          <code>python scripts/start_demo.py</code> from the project root, or{' '}
          <code>uvicorn backend.api.main:app --port 8000</code>. The dot next to the title turns
          green when the models are loaded.
        </li>
        <li>
          <strong>Share the audio.</strong> The browser asks you to pick a tab or window — choose
          the one playing the call and tick &ldquo;share audio&rdquo;. Voice Guard listens to what
          your computer plays, which is where a caller&rsquo;s voice actually arrives.
        </li>
        <li>
          <strong>Tap the button.</strong> Audio is streamed to the backend as 16 kHz PCM over a
          WebSocket. Nothing is recorded.
        </li>
        <li>
          <strong>Wait about four seconds.</strong> The spoof detector needs ~4 s of audio before
          its first verdict, then updates roughly every 3 s while you keep listening.
        </li>
        <li>
          <strong>Read the verdict.</strong> Real, Uncertain, or Synthetic — with the underlying
          0–100 risk score beneath it.
        </li>
        <li>
          <strong>Optionally enrol a reference voice</strong> in Settings. Once enrolled, the system
          also checks whether the speaker is the person you expect, catching a convincing human
          impostor as well as a clone.
        </li>
        <li>
          <strong>Tune it in Settings.</strong> Alert threshold, per-signal weights, and a file test
          mode for recorded samples.
        </li>
      </ol>
    </Section>

    <Section title="What it listens to">
      <p className="prose">
        Voice Guard captures <strong>system or tab audio</strong> — the sound your computer plays.
        That is deliberate and it is the only capture path: a caller&rsquo;s voice reaches you
        through the speaker, so a microphone would record your own side of the conversation plus
        whatever the room adds. Capturing the stream digitally avoids the acoustic path entirely.
      </p>
      <p className="prose">
        It matters for accuracy as well as aim. Room audio commonly sits around 20 dB
        signal-to-noise, below the level where synthetic speech starts evading detection, while
        captured system audio arrives clean and inside the detector&rsquo;s measured range.
      </p>
      <ul className="bullets">
        <li>
          Works for any call running on this computer — Google Meet, Teams, Zoom, WhatsApp Desktop,
          or any browser tab.
        </li>
        <li>
          <strong>A call on your phone cannot be captured.</strong> That audio never reaches this
          computer.
        </li>
        <li>
          The microphone is still used for one thing only: recording a reference voice in Settings.
          It is never a monitoring source.
        </li>
      </ul>
      <p className="hint">
        VoIP calls are often narrowband. If the stream is telephone-band the app declines to judge
        it rather than reporting a verdict it cannot support.
      </p>
    </Section>

    <Section title="Reading the verdict">
      <ul className="verdict-legend">
        <li>
          <ShieldCheck size={16} color="var(--emerald)" />
          <div>
            <strong>Real voice</strong>
            <span>Risk below the suspicion line (35). Speech carries natural human variation.</span>
          </div>
        </li>
        <li>
          <AlertTriangle size={16} color="var(--amber)" />
          <div>
            <strong>Uncertain</strong>
            <span>
              Risk between 35 and your alert threshold. Common with noisy input, very short speech,
              or heavy compression — keep listening.
            </span>
          </div>
        </li>
        <li>
          <ShieldAlert size={16} color="var(--rose)" />
          <div>
            <strong>Synthetic voice</strong>
            <span>Risk at or above the alert threshold (65 by default).</span>
          </div>
        </li>
      </ul>
    </Section>

    <Section title="How the decision is made">
      <p className="prose">
        Three independent signals are computed on every window of audio and fused with explicit,
        adjustable weights — no single black-box classifier owns the verdict.
      </p>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Signal</th>
              <th>Engine</th>
              <th>What it measures</th>
              <th>Weight</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Synthesis detection</td>
              <td>Fine-tuned wav2vec2 (94.6M params)</td>
              <td>Encoder fine-tuned on 19 TTS systems; generalises to unseen generators</td>
              <td className="font-mono">0.70</td>
            </tr>
            <tr>
              <td>Speaker match</td>
              <td>ECAPA-TDNN (SpeechBrain)</td>
              <td>Cosine similarity of a 192-d voiceprint against the enrolled speaker</td>
              <td className="font-mono">0.20</td>
            </tr>
            <tr>
              <td>Acoustic naturalness</td>
              <td>Praat / Parselmouth</td>
              <td>F0 variability, jitter, shimmer, HNR, pause ratio</td>
              <td className="font-mono">0.10</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="formula font-mono">
        risk = ( w<sub>spoof</sub>·(1−bonafide) + w<sub>speaker</sub>·(1−similarity) +
        w<sub>prosody</sub>·(1−naturalness) ) × 100
      </p>
      <p className="hint">
        Each signal is a &ldquo;safe&rdquo; score in [0,1], so it is inverted before weighting. With
        no reference voice enrolled the speaker signal is dropped entirely and its weight is
        redistributed over the other two, so an unused signal never contributes risk of its own —
        for plain synthetic-vs-natural the effective split is 0.875 / 0.125.
      </p>
      <p className="hint">
        Those weights come from measurement rather than intuition. Across every speech-like clip
        tested, the spoof model&rsquo;s output spanned 0.00–0.45 and tracked the label, while the
        prosody score spanned only 0.79–1.00 and pointed the wrong way — it rated genuine TTS
        renders as <em>perfectly</em> natural, because their jitter, shimmer and HNR all fall inside
        the reference bands for human speech. Prosody is therefore kept as a small tie-breaker until
        its ranges are recalibrated against real recordings.
      </p>
    </Section>

    <Section title="Architecture">
      <div className="flow">
        <FlowNode title="System / tab audio" sub="16 kHz mono PCM, streamed from the browser" />
        <Arrow />
        <FlowNode title="Reliability gate" sub="SNR & bandwidth · withholds a verdict out of range" />
        <Arrow />
        <FlowNode title="WebSocket gateway" sub="/ws/analyze · rolling ring buffer, ~4 s window" />
        <Arrow />
        <div className="flow-fanout">
          <FlowNode title="Fine-tuned wav2vec2" sub="synthetic speech detection" />
          <FlowNode title="ECAPA-TDNN" sub="192-d speaker verification" />
          <FlowNode title="Prosody" sub="F0 · jitter · shimmer · HNR" />
        </div>
        <Arrow />
        <FlowNode title="Fusion scorer" sub="weighted, explainable · risk 0–100" />
        <Arrow />
        <FlowNode title="Verdict" sub="real · uncertain · synthetic, with per-signal breakdown" />
      </div>
      <p className="hint">
        Backend: FastAPI + PyTorch, all inference on-device. The spoof model is a fine-tuned
        wav2vec2 encoder running at about 13 ms per 4-second window on a GPU, or 190 ms on CPU —
        the device is detected automatically. Three earlier detectors stay bundled and selectable
        for comparison. Frontend: Next.js 16 with React 19; the browser only captures audio and
        renders state.
      </p>
    </Section>

    <Section title="Streaming behaviour">
      <dl className="kv">
        <div>
          <dt>Sample rate</dt>
          <dd className="font-mono">16 kHz mono</dd>
        </div>
        <div>
          <dt>First verdict after</dt>
          <dd className="font-mono">~4.04 s (64,600 samples)</dd>
        </div>
        <div>
          <dt>Update interval</dt>
          <dd className="font-mono">~3 s</dd>
        </div>
        <div>
          <dt>Rolling buffer</dt>
          <dd className="font-mono">10 s max, oldest dropped</dd>
        </div>
      </dl>
    </Section>

    <Section title="Privacy">
      <ul className="bullets">
        <li>All inference runs locally — no audio leaves your machine for a cloud API.</li>
        <li>Audio lives in a volatile ring buffer and is discarded after each window.</li>
        <li>Enrolment stores only a 192-float embedding, never the recording itself.</li>
        <li>Enrolment is in-memory and clears when the backend restarts.</li>
      </ul>
    </Section>

    <Section title="API">
      <div className="table-scroll">
        <table className="data-table">
          <tbody>
            <tr>
              <td className="font-mono">GET /health</td>
              <td>Model status, device, enrolment state</td>
            </tr>
            <tr>
              <td className="font-mono">POST /analyze</td>
              <td>Analyse an uploaded WAV/FLAC file</td>
            </tr>
            <tr>
              <td className="font-mono">POST /enroll</td>
              <td>Store a reference voiceprint</td>
            </tr>
            <tr>
              <td className="font-mono">DELETE /enroll</td>
              <td>Clear the enrolled speaker</td>
            </tr>
            <tr>
              <td className="font-mono">GET · PUT /config</td>
              <td>Read or update threshold and fusion weights</td>
            </tr>
            <tr>
              <td className="font-mono">WS /ws/analyze</td>
              <td>Live PCM stream in, risk updates out</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Section>

    <Section title="Measured performance">
      <p className="prose">
        Numbers from this build, not from a paper. The spoof model is a wav2vec2 encoder
        fine-tuned on 19 text-to-speech systems, with five generators{' '}
        <strong>held out entirely</strong> so the test measures synthesis it has never seen.
      </p>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>held-out source</th>
              <th>kind</th>
              <th>correct</th>
              <th>n</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Edge-TTS</td>
              <td>synthetic</td>
              <td className="font-mono">100.0%</td>
              <td className="font-mono">161</td>
            </tr>
            <tr>
              <td>ElevenLabs-Turbo-v2.5</td>
              <td>synthetic</td>
              <td className="font-mono">93.1%</td>
              <td className="font-mono">159</td>
            </tr>
            <tr>
              <td>ElevenLabs-v2-Multilingual</td>
              <td>synthetic</td>
              <td className="font-mono">91.7%</td>
              <td className="font-mono">156</td>
            </tr>
            <tr>
              <td>ElevenLabs-v3</td>
              <td>synthetic</td>
              <td className="font-mono">87.2%</td>
              <td className="font-mono">148</td>
            </tr>
            <tr>
              <td>consumer recordings</td>
              <td>genuine</td>
              <td className="font-mono">98.2%</td>
              <td className="font-mono">400</td>
            </tr>
            <tr>
              <td>LibriSpeech, unseen speakers</td>
              <td>genuine</td>
              <td className="font-mono">75.1%</td>
              <td className="font-mono">370</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="hint">
        mean genuine 86.7% · mean synthetic 93.1%. Cross-corpus checks: IEMOCAP / CommonVoice /
        CommonLanguage genuine 100%, older TTS 100%, voice conversion 73.3% caught.
      </p>
      <p className="prose">
        The gain came from fine-tuning the encoder rather than training a classifier on top of it.
        A stock pretrained classifier scored <strong>0%</strong> on these ElevenLabs variants, and
        a head on frozen features reached 69–72% only by dropping genuine accuracy to 55–61%.
        Training the representation itself improved both at once.
      </p>
    </Section>

    <Section title="Never clean up the audio first">
      <p className="prose">
        Noise reduction destroys detection. Measured on five genuine human recordings, denoising
        flipped them from &ldquo;certainly real&rdquo; to &ldquo;certainly fake&rdquo;:
      </p>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>genuine clip</th>
              <th>original</th>
              <th>light NR</th>
              <th>moderate NR</th>
              <th>heavy NR</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>iemo_anger</td>
              <td className="font-mono">1.000</td>
              <td className="font-mono">1.000</td>
              <td className="font-mono">0.000</td>
              <td className="font-mono">0.000</td>
            </tr>
            <tr>
              <td>cv_en</td>
              <td className="font-mono">0.940</td>
              <td className="font-mono">0.000</td>
              <td className="font-mono">0.000</td>
              <td className="font-mono">0.000</td>
            </tr>
            <tr>
              <td>iemo_neutral</td>
              <td className="font-mono">1.000</td>
              <td className="font-mono">1.000</td>
              <td className="font-mono">1.000</td>
              <td className="font-mono">0.001</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="prose">
        Spectral subtraction — what Audacity&rsquo;s Noise Reduction and most &ldquo;enhance
        speech&rdquo; tools do — rewrites the fine spectral detail the model reads, and the model
        scores that processing as synthesis. <strong>Feed it the rawest recording you have.</strong>{' '}
        Reproduce with <code>python scripts/denoise_effect.py</code>.
      </p>
      <p className="hint">
        The app cannot warn you automatically: a spectral-flatness test for denoising artefacts was
        tried and did not separate processed from unprocessed audio, so no such check is shipped
        rather than one that misfires.
      </p>
    </Section>

    <Section title="Global Telemetry & Research Studies">
      <p className="prose">
        Interactive spherical projection study mapping global synthetic voice telemetry and corpus origins.
        Drag to rotate, scroll to zoom, and click to drop telemetry pins.
      </p>
      <div
        style={{
          width: '100%',
          height: '280px',
          borderRadius: 'var(--radius)',
          overflow: 'hidden',
          border: '1px solid var(--border)',
          background: '#08090a',
          marginTop: '12px',
        }}
      >
        <GlobeStudy mode="dark" scale={1} opacity={0.95} />
      </div>
      <p className="hint" style={{ marginTop: '8px' }}>
        Land outlines are rendered from latitude vectors; pin markers indicate recorded sample geographic origin tests.
      </p>
    </Section>

    <Section title="Limits">
      <p className="prose">
        <strong>Clean audiobook narration is the weak spot.</strong> LibriSpeech genuine speech
        scores 75.1% — the lowest figure here, despite being the one genuine corpus present in
        training. Noisier consumer recordings score 98.2%. That inversion is measured but not yet
        explained, so treat verdicts on studio-quality narration with more caution than the
        headline average suggests.
      </p>
      <p className="prose">
        <strong>The generalisation claim has a boundary.</strong> The held-out ElevenLabs clips come
        from the same corpus as the training generators, so they share a recording pipeline. Some of
        that 87–93% may be corpus-level transfer rather than purely generator-level. A clone
        produced independently, through a different pipeline, is the sterner test.
      </p>
      <p className="prose">
        <strong>Voice conversion is harder than text-to-speech.</strong> 73.3% of conversions are
        caught, against 87–100% for TTS. A conversion reuses the original recording&rsquo;s room,
        microphone and channel, so less evidence survives.
      </p>
      <p className="prose">
        Short clips are also weak: under about four seconds the audio is padded to fill the window,
        and genuine speech can flag. Accuracy drops further on noisy rooms, phone-codec audio and
        generators the model has never seen. This is a research and portfolio demo, not a certified
        security control — treat a synthetic verdict as a prompt to verify through another channel,
        never as proof on its own.
      </p>
    </Section>
    </div>
  </div>
);
