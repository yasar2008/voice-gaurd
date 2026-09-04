# Roadmap

> **What a production version would need — and why these were scoped out of the MVP.**

This project demonstrates the technical core: spoof detection + speaker verification + prosody analysis → fused risk score → real-time demo. The items below represent deliberate scoping decisions, not gaps.

---

## Not Built (By Design)

### Telecom / VoIP Integration
- **What**: Tap into SIP trunks, RTP streams, or PBX call legs to analyze live phone calls
- **Why not**: Requires carrier-grade infrastructure (Twilio, Vonage), SBC hardware, and telco compliance — not solo-buildable
- **How to add**: Use Pipecat or FastRTC with a Twilio Media Stream webhook; decode RTP → PCM → existing pipeline

### Multilingual / Accent-Aware Models
- **What**: Fine-tune models for Indian English accents, Hindi, Tamil, and other Indian languages
- **Why not**: Requires curated accent-specific training data and multilingual ASVspoof benchmarks that don't yet exist at scale
- **How to add**: Fine-tune AASIST-L on CommonVoice + accent-labeled data; retrain prosody reference ranges per language

### Historical Fraud Database
- **What**: Cross-session analysis, fraud pattern tracking, known-attacker voiceprint database
- **Why not**: Requires persistent storage, session management, and raises significant privacy concerns
- **How to add**: PostgreSQL + pgvector for embedding storage; build a fraud case management UI

### SMS / Email Alerting
- **What**: Send real-time alerts via SMS (Twilio), email (SendGrid), or push notifications
- **Why not**: Infrastructure concern, not ML — straightforward to add but doesn't demonstrate technical depth
- **How to add**: FastAPI background tasks + Twilio/SendGrid SDK; add alert routing rules

### Banking System SDK
- **What**: Embeddable SDK for banking IVR systems, KYC flows, and transaction verification
- **Why not**: Requires enterprise integration patterns (gRPC, message queues), SLA guarantees, and compliance frameworks
- **How to add**: Package the inference pipeline as a Python/gRPC service; add request authentication and rate limiting

### Compliance Framework
- **What**: GDPR consent flows, India's DPDP Act compliance, data retention policies, audit trails
- **Why not**: Legal/regulatory scope — requires legal review, not engineering
- **How to add**: Add consent management middleware; implement data retention policies; build audit logging

### Edge Deployment / Model Distillation
- **What**: Distill AASIST-L further for mobile/IoT deployment (ONNX, TensorFlow Lite)
- **Why not**: Current model is already lightweight (~85k params); edge deployment needs device-specific optimization
- **How to add**: Export to ONNX → quantize (INT8) → deploy via ONNX Runtime Mobile or TFLite

---

## Planned Improvements (Next Steps)

### Short Term
- [ ] Fine-tune AASIST-L on ASVspoof 2019 LA (currently using random weights for demo)
- [ ] Evaluate on In-the-Wild dataset for generalization metrics
- [ ] Record demo video for README
- [ ] Add WaveFake as secondary training data for improved generalization

### Medium Term
- [ ] Learned fusion layer option (logistic regression) alongside rule-based fusion
- [ ] Voice Activity Detection (VAD) to skip inference during silence
- [ ] WebRTC integration (replace raw WebSocket with proper real-time audio transport)
- [ ] Docker containerization for easy deployment

### Long Term
- [ ] Self-supervised learning features (Wav2Vec 2.0 / HuBERT embeddings as additional fusion signal)
- [ ] Continuous enrollment (update speaker profile over time)
- [ ] A/B testing framework for fusion weight optimization
