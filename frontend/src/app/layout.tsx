import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Voice Guard — AI-Powered Real-Time Voice Clone Detection',
  description:
    'Real-time detection and prevention of voice-cloning impersonation attacks fusing AASIST-L spoof detection, ECAPA-TDNN speaker verification, and clinical prosody heuristics.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
