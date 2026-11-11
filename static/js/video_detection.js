/**
 * SAFETYVISION AI - Live AI Video Analysis Engine
 * Rebuilt Video Detection Flow:
 * - Plays uploaded video directly in a high-performance HTML5 video player with full controls
 * - Concurrently runs YOLOv8 person detection, multi-person tracking, and spatial PPE association
 * - Overlays real-time, flicker-free bounding boxes with explicit PPE labels and confidences
 * - Synchronizes detections to playback timestamps without video freezing or stuttering
 * - Updates Live Side Summary and Person-Wise PPE cards in real time
 * - Debounces state-based safety events to prevent alert spam
 * - Captures representative annotated evidence frame
 * - Automatically stops at video end, renders Final Safety Result, and generates isolated PDF report
 */

class LiveVideoDetectionEngine {
    constructor(config) {
        this.config = config;
        this.videoEl = document.getElementById(config.videoId);
        this.canvasEl = document.getElementById(config.canvasId);
        this.ctx = this.canvasEl ? this.canvasEl.getContext('2d') : null;

        // Container & HUD Elements
        this.viewportContainer = document.getElementById(config.viewportContainerId || 'video-viewport-wrapper');
        this.statusPill = document.getElementById(config.statusPillId || 'video-status-pill');
        this.timePill = document.getElementById(config.timePillId || 'video-time-pill');
        this.fpsPill = document.getElementById(config.fpsPillId || 'video-fps-pill');
        this.latencyPill = document.getElementById(config.latencyPillId || 'video-latency-pill');
        this.errorBanner = document.getElementById('video-error-banner');
        
        // Timeline & Panels
        this.timelineContainer = document.getElementById(config.timelineContainerId || 'video-event-timeline');
        this.currentPeopleContainer = document.getElementById(config.currentPeopleContainerId || 'video-current-people');
        this.finalResultContainer = document.getElementById(config.finalResultContainerId || 'video-final-result-section');
        this.liveSideSummaryContainer = document.getElementById(config.liveSideSummaryContainerId || 'video-live-summary-card');

        // Playback & Analysis State
        this.sessionUid = null;
        this.activeFile = null;
        this.filename = 'uploaded_video.mp4';
        this.duration = 0.0;
        this.isPlaying = false;
        this.isAnalyzing = false;
        this.isCompleted = false;
        this.currentBlobUrl = null;

        // Video-time-deterministic throttle (~5 analyses per video second)
        this.lastAnalyzedVideoTime = -1.0;
        this.minSampleIntervalSec = 0.20;
        this.animFrameId = null;
        this.analysisLoopActive = false;
        this.isSendingFrame = false;

        // Overlay & Stability Buffer (Eliminates blinking)
        this.showBBoxes = true;
        this.showLabels = true;
        this.showHazards = true;
        this.lastRenderedDetections = [];
        this.lastRenderedPersons = [];
        this.emptyGraceCycles = 0;
        this.maxGraceCycles = 3;

        // Final results & evidence
        this.finalData = null;
        this.evidenceSnapshot = null;

        // Offscreen sampling canvas
        this.sampleCanvas = document.createElement('canvas');
        this.sampleCtx = this.sampleCanvas.getContext('2d');

        this.initEvents();
    }

    logVideoDebug(tag, extra = {}) {
        const file = this.activeFile;
        const v = this.videoEl;
        console.log(`[VIDEO DEBUG] =================== ${tag} ===================`);
        console.log(`[VIDEO DEBUG] File selected: ${file ? 'YES' : 'NO'}`);
        console.log(`[VIDEO DEBUG] Filename: ${file ? file.name : (this.filename || 'NONE')}`);
        console.log(`[VIDEO DEBUG] File type: ${file ? (file.type || 'unknown') : 'N/A'}`);
        console.log(`[VIDEO DEBUG] File size: ${file ? (file.size + ' bytes') : 'N/A'}`);
        console.log(`[VIDEO DEBUG] Object URL created: ${this.currentBlobUrl ? 'YES' : 'NO'}`);
        console.log(`[VIDEO DEBUG] Video source assigned: ${v && v.src ? 'YES' : 'NO'}`);
        console.log(`[VIDEO DEBUG] Metadata loaded: ${v && v.readyState >= 1 ? 'YES' : 'NO'}`);
        console.log(`[VIDEO DEBUG] Video width: ${v ? v.videoWidth : 0}`);
        console.log(`[VIDEO DEBUG] Video height: ${v ? v.videoHeight : 0}`);
        console.log(`[VIDEO DEBUG] Duration: ${v ? v.duration : 0}`);
        console.log(`[VIDEO DEBUG] Ready state: ${v ? v.readyState : 0}`);
        console.log(`[VIDEO DEBUG] Can play: ${v && v.readyState >= 3 ? 'YES' : 'NO'}`);
        console.log(`[VIDEO DEBUG] Playing: ${v && !v.paused && !v.ended ? 'YES' : 'NO'}`);
        console.log(`[VIDEO DEBUG] Current time: ${v ? v.currentTime : 0}`);
        console.log(`[VIDEO DEBUG] Video ended: ${v ? (v.ended ? 'YES' : 'NO') : 'NO'}`);
        if (Object.keys(extra).length > 0) {
            console.log(`[VIDEO DEBUG] Extra info:`, extra);
        }
        console.log(`[VIDEO DEBUG] =================================================`);
    }

    initEvents() {
        if (!this.videoEl) return;

        // Metadata loaded event: set duration and align canvas
        this.videoEl.addEventListener('loadedmetadata', () => {
            this.duration = this.videoEl.duration || 0.0;
            this.alignCanvasToVideo();
            this.updateTimeDisplay();
            this.updateStatusPill('READY', '#10B981');
            this.logVideoDebug('loadedmetadata');
        });

        this.videoEl.addEventListener('loadeddata', () => {
            this.alignCanvasToVideo();
            this.logVideoDebug('loadeddata');
        });

        this.videoEl.addEventListener('canplay', () => {
            this.alignCanvasToVideo();
            this.logVideoDebug('canplay');
        });

        // Window & Container resize tracking
        window.addEventListener('resize', () => {
            this.alignCanvasToVideo();
            this.redrawOverlay();
        });

        if (window.ResizeObserver && this.viewportContainer) {
            const ro = new ResizeObserver(() => {
                this.alignCanvasToVideo();
                this.redrawOverlay();
            });
            ro.observe(this.viewportContainer);
        }

        // Native playback events
        this.videoEl.addEventListener('play', () => {
            this.isPlaying = true;
            this.isCompleted = false;
            this.updateStatusPill('PLAYING', '#10B981');
            this.updatePlayPauseButton(true);
            this.startAnalysisLoop();
            // Visual only: show the subtle scan line while genuinely playing
            if (this.viewportContainer) this.viewportContainer.classList.add('scanning');
            this.logVideoDebug('play');
        });

        this.videoEl.addEventListener('pause', () => {
            this.isPlaying = false;
            if (!this.isCompleted) {
                this.updateStatusPill('PAUSED', '#F59E0B');
            }
            // Visual only: hide the scan line when not playing
            if (this.viewportContainer) this.viewportContainer.classList.remove('scanning');
            this.updatePlayPauseButton(false);
            this.logVideoDebug('pause');
        });

        this.videoEl.addEventListener('ended', () => {
            console.log('[VIDEO] Playback ended naturally. Finalizing session...');
            this.isPlaying = false;
            this.stopAnalysisLoop();
            this.updatePlayPauseButton(false);
            this.updateStatusPill('ANALYSIS COMPLETE', '#10B981');
            // Visual only: hide the scan line once playback ends
            if (this.viewportContainer) this.viewportContainer.classList.remove('scanning');
            this.logVideoDebug('ended');
            this.onVideoEnded();
        });

        this.videoEl.addEventListener('timeupdate', () => {
            this.updateTimeDisplay();
            this.updateScrubber();
        });

        // Diagnostics for network/decoding issues
        this.videoEl.addEventListener('error', (e) => {
            const err = this.videoEl.error;
            console.error('[VIDEO DEBUG] Video element error:', err, e);
            this.logVideoDebug('error', { code: err ? err.code : null, message: err ? err.message : null });
            this.showVideoError("Unable to play this video in the browser. Please verify the video is an H.264 MP4, WebM, or supported browser format.");
        });

        this.videoEl.addEventListener('stalled', () => this.logVideoDebug('stalled'));
        this.videoEl.addEventListener('abort', () => this.logVideoDebug('abort'));
        this.videoEl.addEventListener('emptied', () => this.logVideoDebug('emptied'));
    }

    alignCanvasToVideo() {
        if (!this.videoEl || !this.canvasEl || !this.viewportContainer) return;
        const vw = this.videoEl.videoWidth || 0;
        const vh = this.videoEl.videoHeight || 0;
        if (vw === 0 || vh === 0) return;

        // Set native canvas buffer dimensions matching video resolution
        if (this.canvasEl.width !== vw || this.canvasEl.height !== vh) {
            this.canvasEl.width = vw;
            this.canvasEl.height = vh;
        }
        if (this.sampleCanvas.width !== vw || this.sampleCanvas.height !== vh) {
            this.sampleCanvas.width = vw;
            this.sampleCanvas.height = vh;
        }

        // The video element itself fills the wrapper (object-fit: contain
        // letterboxes the VIDEO CONTENT inside it). So the real displayed
        // content rectangle must be computed from the WRAPPER, not from the
        // video element's own rect (which is already the full wrapper box).
        // Re-letterboxing the element rect would shrink + shift the overlay.
        const cRect = this.viewportContainer.getBoundingClientRect();
        const availW = cRect.width;
        const availH = cRect.height;
        if (availW <= 0 || availH <= 0) return;

        const videoRatio = vw / vh;
        const wrapRatio = availW / availH;

        let displayW, displayH, offsetX, offsetY;
        if (wrapRatio > videoRatio) {
            // Wrapper wider than video -> pillarbox (bars left/right)
            displayH = availH;
            displayW = displayH * videoRatio;
            offsetX = (availW - displayW) / 2;
            offsetY = 0;
        } else {
            // Wrapper taller than (or equal to) video -> letterbox (bars top/bottom)
            displayW = availW;
            displayH = displayW / videoRatio;
            offsetX = 0;
            offsetY = (availH - displayH) / 2;
        }

        this.canvasEl.style.position = 'absolute';
        this.canvasEl.style.left = `${Math.round(offsetX)}px`;
        this.canvasEl.style.top = `${Math.round(offsetY)}px`;
        this.canvasEl.style.width = `${Math.round(displayW)}px`;
        this.canvasEl.style.height = `${Math.round(displayH)}px`;
    }

    async loadVideoFile(file) {
        if (!file) return;

        this.hideVideoError();
        this.activeFile = file;
        this.filename = file.name || 'uploaded_video.mp4';
        this.isCompleted = false;
        this.finalData = null;
        this.evidenceSnapshot = null;
        this.lastRenderedPersons = [];
        this.lastRenderedDetections = [];
        this.emptyGraceCycles = 0;
        this.duration = 0.0;
        this.lastAnalyzedVideoTime = -1.0;
        this.sessionUid = null;

        // Reset UI Panels
        this.resetUI();
        // Visual only: new file loaded means idle until the user presses play
        if (this.viewportContainer) this.viewportContainer.classList.remove('scanning');
        if (this.ctx && this.canvasEl) {
            this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
        }

        // Set initial waiting HUD states
        if (this.timePill) this.timePill.textContent = '00:00 / --:--';
        if (this.fpsPill) this.fpsPill.textContent = 'FPS: —';
        if (this.latencyPill) this.latencyPill.textContent = 'Latency: —';
        this.updateStatusPill('READY', '#10B981');

        // Revoke previous blob URL if needed
        if (this.currentBlobUrl) {
            try { URL.revokeObjectURL(this.currentBlobUrl); } catch(e) {}
        }

        // Create Object URL for instant local video playback
        try {
            this.currentBlobUrl = URL.createObjectURL(file);
        } catch(err) {
            console.error('[VIDEO DEBUG] Failed to create object URL:', err);
            this.showVideoError("Unable to open selected video file in browser.");
            return;
        }

        this.videoEl.src = this.currentBlobUrl;
        this.videoEl.preload = 'auto';
        this.videoEl.load();

        this.logVideoDebug('File selected');

        // Initialize backend session in parallel (non-blocking for player)
        this.initBackendSession();
    }

    async initBackendSession() {
        try {
            const resp = await fetch('/api/video/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content },
                body: JSON.stringify({
                    filename: this.filename,
                    duration_seconds: this.duration || 0.0
                })
            });
            const data = await resp.json();
            if (data.success && data.session_uid) {
                this.sessionUid = data.session_uid;
                console.log(`[VIDEO] Backend session initialized: ${this.sessionUid}`);
                this.updateSessionLabels();
            }
        } catch (err) {
            console.warn('[VIDEO] Backend session init fallback:', err);
            this.sessionUid = 'VID-' + new Date().toISOString().slice(0,10).replace(/-/g,'') + '-001';
        }
    }

    async playPause() {
        if (!this.videoEl) return;

        if (this.videoEl.paused) {
            if (!this.videoEl.src) {
                this.showVideoError("No video loaded. Please select a video file first.");
                return;
            }

            // Verify playable readyState and non-zero dimensions
            if (this.videoEl.readyState < 2 || !this.videoEl.videoWidth || !this.videoEl.videoHeight) {
                console.warn('[VIDEO DEBUG] Video not ready yet. readyState:', this.videoEl.readyState, 'Width:', this.videoEl.videoWidth);
                this.showVideoError("Video is still loading. Please wait...");
                return;
            }

            this.hideVideoError();

            // Ensure session exists
            if (!this.sessionUid) {
                await this.initBackendSession();
            }

            try {
                await this.videoEl.play();
            } catch (err) {
                console.error('[VIDEO DEBUG] play() error:', err);
                this.showVideoError("Playback failed: " + (err.message || "Browser prevented video playback."));
            }
        } else {
            this.videoEl.pause();
        }
    }

    startAnalysisLoop() {
        if (this.analysisLoopActive) return;
        this.analysisLoopActive = true;
        this.isAnalyzing = true;
        this.lastAnalyzedVideoTime = -1.0;

        const frameTick = async () => {
            if (!this.analysisLoopActive) return;

            if (this.videoEl && !this.videoEl.paused && !this.videoEl.ended) {
                const curTime = this.videoEl.currentTime || 0.0;
                
                // Deterministic sampling: sample frames based on video timeline progression
                const timeDiff = curTime - this.lastAnalyzedVideoTime;
                if ((this.lastAnalyzedVideoTime < 0 || timeDiff >= this.minSampleIntervalSec || timeDiff < -0.5) && !this.isSendingFrame) {
                    this.lastAnalyzedVideoTime = curTime;
                    await this.captureAndAnalyzeCurrentFrame();
                }
            }

            if (this.analysisLoopActive) {
                this.animFrameId = requestAnimationFrame(frameTick);
            }
        };

        this.animFrameId = requestAnimationFrame(frameTick);
    }

    stopAnalysisLoop() {
        this.analysisLoopActive = false;
        this.isAnalyzing = false;
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
    }

    async captureAndAnalyzeCurrentFrame() {
        if (!this.videoEl || this.videoEl.paused || this.videoEl.ended) return;
        const vw = this.videoEl.videoWidth;
        const vh = this.videoEl.videoHeight;
        if (vw === 0 || vh === 0) return;

        this.alignCanvasToVideo();

        this.sampleCanvas.width = vw;
        this.sampleCanvas.height = vh;
        this.sampleCtx.drawImage(this.videoEl, 0, 0, vw, vh);

        const dataUrl = this.sampleCanvas.toDataURL('image/jpeg', 0.65);
        const curTime = this.videoEl.currentTime || 0.0;

        this.isSendingFrame = true;
        this.updateStatusPill('AI ANALYZING', '#38BDF8');

        try {
            const response = await fetch('/api/video/analyze_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content },
                body: JSON.stringify({
                    session_uid: this.sessionUid,
                    image_data: dataUrl,
                    video_time: curTime
                })
            });

            if (response.ok) {
                const result = await response.json();
                if (result.success) {
                    // Render Stable Bounding Boxes on Overlay Canvas
                    this.renderOverlay(result);

                    // Update HUD Stats (FPS & Latency)
                    if (this.fpsPill) this.fpsPill.textContent = `${result.fps || 0.0} FPS`;
                    if (this.latencyPill) this.latencyPill.textContent = `${result.latency_ms || 0.0} ms`;

                    // Update Live Side Panel (People & Compliance)
                    this.updateLiveSidePanel(result);

                    // Update Person-Wise Independent Cards
                    this.updateCurrentPeopleCards(result.person_results || []);

                    // Append New Timeline Events (Zero spam, state changes only)
                    if (result.new_events && result.new_events.length > 0) {
                        result.new_events.forEach(evt => this.appendTimelineEvent(evt));
                    }

                    // Track local evidence screenshot candidate
                    const persons = result.person_results || [];
                    if (persons.length > 0 && !this.evidenceSnapshot) {
                        this.evidenceSnapshot = dataUrl;
                    } else if (persons.some(p => p.status !== 'COMPLIANT')) {
                        this.evidenceSnapshot = dataUrl;
                    }
                }
            }
        } catch (err) {
            console.warn('[VIDEO DEBUG] Frame analysis error:', err);
        } finally {
            this.isSendingFrame = false;
            if (this.isPlaying && !this.videoEl.paused && !this.videoEl.ended) {
                this.updateStatusPill('PLAYING', '#10B981');
            }
        }
    }

    renderOverlay(analysis) {
        if (!this.ctx) return;
        let persons = analysis.person_results || [];
        let rawDets = analysis.raw_detections || [];

        // Stability Grace Buffer: if model momentarily drops 1 frame, retain boxes to prevent blinking
        if (persons.length === 0) {
            if (this.emptyGraceCycles < this.maxGraceCycles && this.lastRenderedPersons.length > 0) {
                persons = this.lastRenderedPersons;
                rawDets = this.lastRenderedDetections;
                this.emptyGraceCycles++;
            } else {
                this.emptyGraceCycles = 0;
                this.lastRenderedPersons = [];
                this.lastRenderedDetections = [];
            }
        } else {
            this.emptyGraceCycles = 0;
            this.lastRenderedPersons = persons;
            this.lastRenderedDetections = rawDets;
        }

        this.drawScene(persons, rawDets);
    }

    redrawOverlay() {
        if (!this.ctx) return;
        this.drawScene(this.lastRenderedPersons, this.lastRenderedDetections);
    }

    drawScene(persons, rawDets) {
        const cw = this.canvasEl.width;
        const ch = this.canvasEl.height;
        this.ctx.clearRect(0, 0, cw, ch);

        if (!this.showBBoxes) return;

        // Dynamic Resolution-Independent Scaling Factor
        // Ensures labels remain MEDIUM and perfectly readable on 4K, 1080p, or 720p
        const displayW = parseFloat(this.canvasEl.style.width) || this.canvasEl.clientWidth || cw;
        const scale = Math.max(1.0, cw / Math.max(displayW, 100));

        const fontPerson = `bold ${Math.round(14 * scale)}px Inter, system-ui, sans-serif`;
        const fontPPE = `bold ${Math.round(11 * scale)}px Inter, system-ui, sans-serif`;
        const fontMissing = `bold ${Math.round(11 * scale)}px Inter, system-ui, sans-serif`;
        const linePerson = Math.max(2, Math.round(2.5 * scale));
        const linePPE = Math.max(1.5, Math.round(1.8 * scale));
        const padX = Math.round(6 * scale);
        const padY = Math.round(4 * scale);
        const personHeaderH = Math.round(26 * scale);
        const ppeTagH = Math.round(18 * scale);

        // Semi-transparent label background color (white with alpha for readability)
        const labelBg = 'rgba(255, 255, 255, 0.88)';
        const labelTextColor = '#111827';

        const drawnPPEBoxes = new Set();

        // 1. Draw Environmental Hazards (Machinery, Vehicles, Cones)
        if (this.showHazards && rawDets) {
            rawDets.forEach(det => {
                const cname = det.class_name;
                if (!cname || cname === 'Person' || cname.includes('Hardhat') || cname.includes('Mask') || cname.includes('Vest') || !det.bbox) return;

                const [x1, y1, x2, y2] = det.bbox;
                const conf = det.confidence ? Math.round(det.confidence * 100) : 80;
                const color = det.color || (cname === 'Safety Cone' ? '#F59E0B' : (cname === 'machinery' ? '#EC4899' : '#6366F1'));

                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = linePPE;
                this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                const tag = `${cname} ${conf}%`;
                this.ctx.font = fontPPE;
                const tw = this.ctx.measureText(tag).width;
                const tagY = Math.max(ppeTagH + y1, Math.min(ch - ppeTagH, y1 + 2));
                const tH = ppeTagH;

                // Compact semi-transparent label background behind text only
                this.ctx.fillStyle = labelBg;
                this.ctx.fillRect(x1, tagY - tH, tw + padX * 2, tH);
                this.ctx.fillStyle = labelTextColor;
                this.ctx.fillText(tag, x1 + padX, tagY - padY);
            });
        }

        // 2. Draw Associated PPE Bounding Boxes (Explicit Class Names & Confidences)
        persons.forEach(person => {
            const pBox = person.bbox;
            if (pBox) {
                const [px1, py1, px2, py2] = pBox;
                const pColor = person.isViolation ? '#EF4444' : (person.risk === 'CRITICAL' ? '#EF4444' : (person.risk === 'HIGH' ? '#F97316' : '#10B981'));
                this.ctx.strokeStyle = pColor;
                this.ctx.lineWidth = linePerson;
                this.ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
            }

            const items = person.associated_ppe || [];
            items.forEach(it => {
                if (!it.bbox) return;
                const [ix1, iy1, ix2, iy2] = it.bbox;
                const boxKey = `${Math.round(ix1)}_${Math.round(iy1)}_${Math.round(ix2)}_${Math.round(iy2)}`;
                drawnPPEBoxes.add(boxKey);

                const cname = it.class_name || 'PPE';
                const conf = it.confidence ? Math.round(it.confidence * 100) : 85;
                const isNeg = cname.startsWith('NO-');

                let boxColor = '#10B981';
                if (isNeg) {
                    boxColor = (cname.includes('Safety Vest') ? '#DC2626' : (cname.includes('Hardhat') ? '#EF4444' : '#F97316'));
                } else {
                    boxColor = (cname.includes('Hardhat') ? '#10B981' : (cname.includes('Mask') ? '#06B6D4' : '#3B82F6'));
                }

                this.ctx.strokeStyle = boxColor;
                this.ctx.lineWidth = linePPE;
                this.ctx.strokeRect(ix1, iy1, ix2 - ix1, iy2 - iy1);

                if (this.showLabels) {
                    const tag = `${cname} ${conf}%`;
                    this.ctx.font = fontPPE;
                    const tw = this.ctx.measureText(tag).width;
                    const tagY = Math.max(iy1 - ppeTagH, 4);
                    const tH = ppeTagH;

                    // Compact semi-transparent label background behind text only
                    this.ctx.fillStyle = labelBg;
                    this.ctx.fillRect(ix1, tagY - tH, tw + padX * 2, tH);
                    this.ctx.fillStyle = labelTextColor;
                    this.ctx.fillText(tag, ix1 + padX, tagY - padY);
                }
            });
        });

        // 3. Draw Unassociated PPE Detections
        if (rawDets) {
            rawDets.forEach(det => {
                const cname = det.class_name;
                if (!cname || cname === 'Person' || !det.bbox) return;
                if (!cname.includes('Hardhat') && !cname.includes('Mask') && !cname.includes('Vest')) return;

                const [ix1, iy1, ix2, iy2] = det.bbox;
                const boxKey = `${Math.round(ix1)}_${Math.round(iy1)}_${Math.round(ix2)}_${Math.round(iy2)}`;
                if (drawnPPEBoxes.has(boxKey)) return;

                const conf = det.confidence ? Math.round(det.confidence * 100) : 80;
                const isNeg = cname.startsWith('NO-');
                const boxColor = isNeg ? '#EF4444' : '#10B981';

                this.ctx.strokeStyle = boxColor;
                this.ctx.lineWidth = linePPE;
                this.ctx.strokeRect(ix1, iy1, ix2 - ix1, iy2 - iy1);

                if (this.showLabels) {
                    const tag = `${cname} ${conf}%`;
                    this.ctx.font = fontPPE;
                    const tw = this.ctx.measureText(tag).width;
                    const tagY = Math.max(iy1 - ppeTagH, 4);
                    const tH = ppeTagH;

                    // Compact semi-transparent label background behind text only
                    this.ctx.fillStyle = labelBg;
                    this.ctx.fillRect(ix1, tagY - tH, tw + padX * 2, tH);
                    this.ctx.fillStyle = labelTextColor;
                    this.ctx.fillText(tag, ix1 + padX, tagY - padY);
                }
            });
        }

        // 4. Draw Person Outer Bounding Boxes with Visual Hierarchy & Clear Medium Labels
        persons.forEach(person => {
            if (!person.bbox) return;
            const [x1, y1, x2, y2] = person.bbox;
            const w = x2 - x1;
            const h = y2 - y1;

            const isCompliant = (person.status === 'COMPLIANT');
            const risk = person.risk || (isCompliant ? 'LOW' : 'MEDIUM');

            let color = '#10B981'; // Green (Safe)
            if (risk === 'CRITICAL') color = '#EF4444';
            else if (risk === 'HIGH') color = '#F97316';
            else if (!isCompliant) color = '#F59E0B';

            // Person outer box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = linePerson;
            this.ctx.strokeRect(x1, y1, w, h);

            // Corner highlights
            const clen = Math.min(24 * scale, Math.min(w, h) * 0.2);
            this.ctx.lineWidth = linePerson + 1.5;
            // Top-left
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1 + clen);
            this.ctx.lineTo(x1, y1);
            this.ctx.lineTo(x1 + clen, y1);
            this.ctx.stroke();

            // Top-right
            this.ctx.beginPath();
            this.ctx.moveTo(x2 - clen, y1);
            this.ctx.lineTo(x2, y1);
            this.ctx.lineTo(x2, y1 + clen);
            this.ctx.stroke();

            // Medium, professional Person Header Pill
            const label = person.label || `Person #${person.person_id || 1}`;
            const statusText = isCompliant ? 'COMPLIANT' : 'PPE VIOLATION';
            const tag = `${label} | ${statusText}`;

            this.ctx.font = fontPerson;
            const tagW = this.ctx.measureText(tag).width;

            this.ctx.fillStyle = color;
                this.ctx.fillRect(x1, Math.max(0, y1 - personHeaderH), tagW + padX * 2, personHeaderH);
                this.ctx.fillStyle = labelTextColor;
                this.ctx.fillText(tag, x1 + padX, Math.max(personHeaderH - padY * 1.5, y1 - padY * 1.5));

            // Bottom Missing Summary Strip (if violating)
            if (!isCompliant && person.missing_items && person.missing_items.length > 0) {
                const missingText = `Missing: ${person.missing_items.join(', ')}`;
                this.ctx.font = fontMissing;
                const mw = this.ctx.measureText(missingText).width;
                const by = Math.min(ch - 4, y2 + ppeTagH);

                this.ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
                this.ctx.fillRect(x1, by - ppeTagH, mw + padX * 2, ppeTagH);

                this.ctx.fillStyle = '#FCA5A5';
                this.ctx.fillText(missingText, x1 + padX, by - padY);
            }
        });
    }

    updateLiveSidePanel(result) {
        const sum = result.live_summary || {};
        const elPeople = document.getElementById('kpi-video-people');
        const elCompliant = document.getElementById('kpi-video-compliant');
        const elViolations = document.getElementById('kpi-video-violations');
        const elCritical = document.getElementById('kpi-video-critical');
        const elComplianceRate = document.getElementById('kpi-video-compliance-rate');
        const elStatusBanner = document.getElementById('video-live-status-banner');

        if (elPeople) elPeople.textContent = sum.people_detected || 0;
        if (elCompliant) elCompliant.textContent = sum.compliant || 0;
        // Canonical: PPE Violations = individually missing items.
        if (elViolations) { elViolations.textContent = (sum.ppe_violations !== undefined ? sum.ppe_violations : sum.violations) || 0; elViolations.title = 'Individually missing PPE items'; }
        if (elCritical) elCritical.textContent = sum.critical || 0;

        const rateStr = (sum.people_detected > 0 && sum.overall_compliance !== undefined && sum.overall_compliance !== 'N/A')
            ? `${sum.overall_compliance}%`
            : 'N/A';
        if (elComplianceRate) elComplianceRate.textContent = rateStr;

        if (elStatusBanner) {
            const itemViol = (sum.ppe_violations !== undefined ? sum.ppe_violations : sum.violations) || 0;
            const nonComp = (sum.non_compliant_people !== undefined) ? sum.non_compliant_people : null;
            elStatusBanner.style.display = 'block';
            if (itemViol === 0 && (sum.people_detected || 0) > 0) {
                elStatusBanner.style.background = 'var(--safe-subtle)';
                elStatusBanner.style.color = 'var(--safe-text)';
                elStatusBanner.style.border = '1px solid var(--safe-border)';
                elStatusBanner.textContent = '✓ ALL VISIBLE PERSONNEL COMPLIANT';
            } else if (itemViol > 0) {
                elStatusBanner.style.background = 'var(--critical-subtle)';
                elStatusBanner.style.color = 'var(--critical-text)';
                elStatusBanner.style.border = '1px solid var(--critical-border)';
                elStatusBanner.textContent = nonComp !== null
                    ? `⚠ ${itemViol} PPE VIOLATION${itemViol === 1 ? '' : 'S'} ACROSS ${nonComp} WORKER${nonComp === 1 ? '' : 'S'}`
                    : `⚠ ${itemViol} PPE VIOLATION${itemViol === 1 ? '' : 'S'}`;
            } else {
                elStatusBanner.style.background = 'var(--bg-card-subtle)';
                elStatusBanner.style.color = 'var(--text-muted)';
                elStatusBanner.style.border = '1px solid var(--border-subtle)';
                elStatusBanner.textContent = 'MONITORING VIDEO STREAM';
            }
        }
    }

    updateCurrentPeopleCards(personResults) {
        if (!this.currentPeopleContainer) return;

        if (!personResults || personResults.length === 0) {
            this.currentPeopleContainer.innerHTML = `
                <div style="text-align:center; padding:28px 14px; color:var(--text-muted); background:var(--bg-card-subtle); border-radius:var(--radius-md);">
                    <div style="margin-bottom:6px; font-size:1.4rem;">👤</div>
                    <div style="font-weight:600; font-size:0.85rem; color:var(--text-primary); margin-bottom:2px;">No Personnel in Current Frame</div>
                    <div style="font-size:0.75rem;">Personnel will be analyzed independently as they appear.</div>
                </div>
            `;
            return;
        }

        let html = '';
        personResults.forEach(p => {
            const compVal = (p.compliance_score !== undefined) ? p.compliance_score : 100.0;
            const isCompliant = (p.status === 'COMPLIANT');
            const risk = p.risk || (isCompliant ? 'LOW' : 'MEDIUM');
            const borderColor = isCompliant ? '#10B981' : (risk === 'CRITICAL' ? '#EF4444' : (risk === 'HIGH' ? '#F97316' : '#F59E0B'));
            const statusBadgeClass = isCompliant ? 'badge-low' : (risk === 'CRITICAL' ? 'badge-critical' : 'badge-high');
            const statusText = p.status || (isCompliant ? 'COMPLIANT' : 'PPE VIOLATION');

            const renderGearRow = (gearName, val, labelOverride, conf) => {
                const isPresent = (val === 'PRESENT');
                const badgeColor = isPresent ? '#10B981' : '#DC2626';
                const badgeText = labelOverride || (isPresent ? '✓ DETECTED' : '✗ MISSING');
                const confPercent = (conf > 1.0) ? Math.round(conf) : Math.round(conf * 100);
                const confTag = (conf && conf > 0) ? `<span style="font-size:0.72rem; color:var(--text-muted); margin-left:4px;">(${confPercent}%)</span>` : '';
                return `
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid rgba(0,0,0,0.04);">
                        <span style="font-size:0.82rem; color:var(--text-secondary);">${gearName}</span>
                        <div>
                            <span style="color:${badgeColor}; font-weight:700; font-size:0.8rem;">${badgeText}</span>
                            ${confTag}
                        </div>
                    </div>
                `;
            };

            html += `
                <div class="card" style="margin-bottom:12px; border-left:4px solid ${borderColor}; box-shadow:var(--shadow-sm); overflow:hidden;">
                    <div style="padding:10px 14px; background:var(--bg-card-subtle); display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-subtle);">
                        <div style="font-family:var(--font-display); font-weight:700; font-size:0.95rem; color:var(--text-primary);">
                            ${p.label || ('Person #' + p.person_id)}
                        </div>
                        <span class="badge ${statusBadgeClass}">${statusText}</span>
                    </div>
                    <div style="padding:10px 14px; background:var(--bg-card);">
                        ${renderGearRow('Hardhat', p.hardhat, p.hardhat_label, p.hardhat_conf)}
                        ${renderGearRow('Protective Mask', p.mask, p.mask_label, p.mask_conf)}
                        ${renderGearRow('Safety Vest', p.vest, p.vest_label, p.vest_conf)}

                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; padding-top:8px; border-top:1px solid var(--border-subtle); font-size:0.78rem;">
                            <span>Compliance: <b style="color:${isCompliant ? '#10B981' : '#DC2626'}; font-size:0.9rem;">${compVal}%</b></span>
                            <span>Risk: <b style="color:${borderColor};">${risk}</b></span>
                        </div>
                    </div>
                </div>
            `;
        });

        this.currentPeopleContainer.innerHTML = html;
    }

    appendTimelineEvent(evt) {
        if (!this.timelineContainer) return;

        // Remove placeholder if present
        const emptyNotice = this.timelineContainer.querySelector('.timeline-empty-notice');
        if (emptyNotice) emptyNotice.remove();

        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.style.cssText = 'display:flex; gap:12px; padding:8px 0; border-bottom:1px solid var(--border-subtle); font-size:0.82rem; align-items:flex-start;';

        const timeBadgeColor = evt.badge_color || (evt.event_type === 'COMPLIANCE_RESTORED' ? '#10B981' : '#EF4444');

        item.innerHTML = `
            <div style="font-family:var(--font-mono); font-weight:700; font-size:0.75rem; color:${timeBadgeColor}; background:rgba(0,0,0,0.05); padding:2px 6px; border-radius:4px; white-space:nowrap;">
                ${evt.timestamp_str || '00:00'}
            </div>
            <div style="flex:1;">
                <div style="font-weight:700; color:var(--text-primary);">
                    ${evt.title || evt.label}
                </div>
                <div style="color:var(--text-secondary); font-size:0.78rem; margin-top:1px;">
                    ${evt.description || ''}
                </div>
            </div>
        `;

        this.timelineContainer.appendChild(item);
        this.timelineContainer.scrollTop = this.timelineContainer.scrollHeight;
    }

    async onVideoEnded() {
        if (this.isCompleted) return;
        this.isCompleted = true;

        this.updateStatusPill('FINALIZING REPORT...', '#38BDF8');
        console.log('[VIDEO] Sending session completion request...');

        try {
            const resp = await fetch('/api/video/complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content },
                body: JSON.stringify({
                    session_uid: this.sessionUid,
                    duration_seconds: this.videoEl.currentTime || this.duration,
                    filename: this.filename,
                    evidence_image: this.evidenceSnapshot || ''
                })
            });

            if (resp.ok) {
                const finalData = await resp.json();
                if (finalData.success) {
                    this.finalData = finalData;
                    this.updateStatusPill('ANALYSIS COMPLETE', '#10B981');
                    this.renderFinalSafetyResult(finalData);
                    if (typeof showToast === 'function') {
                        showToast('Video Analysis Complete', `Session ${finalData.session_uid} finalized. People: ${finalData.people_observed}, Overall compliance: ${finalData.overall_compliance}%`, 'SUCCESS');
                    }
                    // Reflect the finalized session in the Recent Analysis section (detection center)
                    if (typeof refreshRecentAnalysis === 'function') {
                        refreshRecentAnalysis();
                    }
                }
            }
        } catch (err) {
            console.error('[VIDEO] Error completing video session:', err);
            this.updateStatusPill('ANALYSIS FINISHED', '#10B981');
        }
    }

    renderFinalSafetyResult(data) {
        if (!this.finalResultContainer) return;

        this.finalResultContainer.style.display = 'block';
        this.finalResultContainer.scrollIntoView({ behavior: 'smooth' });

        const statusColor = data.safety_badge_color || (data.safety_status === 'SAFE' ? '#10B981' : (data.safety_status === 'UNSAFE' ? '#EF4444' : '#F59E0B'));
        const ppe = data.ppe_breakdown || {};

        // Render Person-Wise Cards in Final Report
        let personCardsHtml = '';
        const allPersons = data.person_results || [];

        if (allPersons.length === 0) {
            personCardsHtml = `
                <div style="text-align:center; padding:24px; color:var(--text-muted); background:var(--bg-card-subtle); border-radius:var(--radius-md);">
                    No personnel were observed throughout this video playback.
                </div>
            `;
        } else {
            allPersons.forEach(p => {
                const comp = (p.compliance_score !== undefined) ? p.compliance_score : 100.0;
                const isComp = (p.status === 'COMPLIANT');
                const pColor = isComp ? '#10B981' : (p.risk === 'CRITICAL' ? '#EF4444' : (p.risk === 'HIGH' ? '#F97316' : '#F59E0B'));
                const pTextColor = isComp ? 'var(--safe)' : (p.risk === 'CRITICAL' ? 'var(--critical)' : (p.risk === 'HIGH' ? 'var(--high)' : 'var(--warning)'));

                const renderCheck = (val) => val === 'PRESENT'
                    ? '<span style="color:var(--safe); font-weight:700;">✓ DETECTED</span>'
                    : '<span style="color:var(--critical); font-weight:700;">✗ MISSING</span>';

                personCardsHtml += `
                    <div class="card" style="margin-bottom:12px; border-left:4px solid ${pColor};">
                        <div style="padding:10px 14px; background:var(--bg-card-subtle); display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-subtle);">
                            <div style="font-weight:700; font-size:0.95rem; color:var(--text-primary); font-family:var(--font-display);">
                                ${p.label || ('Person #' + p.person_id)}
                            </div>
                            <span class="badge" style="background:${isComp ? 'var(--safe-subtle)' : 'var(--critical-subtle)'}; color:${isComp ? 'var(--safe)' : 'var(--critical)'}; border:1px solid ${pColor}40;">
                                ${p.status}
                            </span>
                        </div>
                        <div style="padding:12px 14px; background:var(--bg-card); display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; font-size:0.8rem;">
                            <div>
                                <span style="color:var(--text-muted);">Hardhat:</span><br/>
                                ${renderCheck(p.hardhat)}
                            </div>
                            <div>
                                <span style="color:var(--text-muted);">Mask:</span><br/>
                                ${renderCheck(p.mask)}
                            </div>
                            <div>
                                <span style="color:var(--text-muted);">Safety Vest:</span><br/>
                                ${renderCheck(p.vest)}
                            </div>
                        </div>
                        <div style="padding:8px 14px; background:var(--bg-card-subtle); border-top:1px solid var(--border-subtle); display:flex; justify-content:space-between; font-size:0.78rem;">
                            <span>Individual Compliance: <b style="color:${pTextColor}; font-size:0.88rem;">${comp}%</b></span>
                            <span>Assessed Risk: <b style="color:${pTextColor};">${p.risk || 'LOW'}</b></span>
                        </div>
                    </div>
                `;
            });
        }

        // Render Recommendations
        let recsHtml = '';
        const recs = data.recommendations || [];
        recs.forEach(r => {
            recsHtml += `
                <div style="padding:10px 14px; background:var(--bg-card-subtle); border-left:4px solid ${r.color || '#2563EB'}; border-radius:4px; margin-bottom:8px;">
                    <div style="font-weight:700; font-size:0.85rem; color:var(--text-primary);">${r.title}</div>
                    <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:2px;">${r.action}</div>
                </div>
            `;
        });

        const overallCompStr = (data.overall_compliance === 'N/A' || data.people_observed === 0)
            ? 'N/A'
            : `${data.overall_compliance}%`;

        this.finalResultContainer.innerHTML = `
            <div class="card" style="margin-top:24px; border:2px solid ${statusColor}; box-shadow:var(--shadow-md);">
                <!-- Completion Banner -->
                <div style="background:#0B1120; color:#FFFFFF; padding:16px 20px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; border-top-left-radius:var(--radius-lg); border-top-right-radius:var(--radius-lg);">
                    <div>
                        <div style="font-size:0.75rem; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:0.05em;">
                            ✓ VIDEO ANALYSIS COMPLETE
                        </div>
                        <h2 style="font-family:var(--font-display); font-size:1.3rem; font-weight:700; margin:2px 0 0; color:#FFFFFF;">
                            Final Workplace Safety Evaluation
                        </h2>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="badge" style="background:${statusColor}; color:#FFFFFF; font-size:0.85rem; padding:6px 14px; font-weight:700;">
                            STATUS: ${data.safety_status}
                        </span>
                    </div>
                </div>

                <div class="card-body" style="padding:24px;">
                    <!-- Top Metrics 5-Column Grid (Traceable to Person-Wise Results) -->
                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:12px; margin-bottom:24px;">
                        <div style="background:var(--bg-card-subtle); padding:12px; border-radius:var(--radius-md); border:1px solid var(--border-subtle);">
                            <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">VIDEO DURATION</div>
                            <div style="font-size:1.3rem; font-weight:700; color:var(--text-primary); font-family:var(--font-mono);">${data.duration_formatted || (data.duration_seconds + 's')}</div>
                        </div>
                        <div style="background:var(--bg-card-subtle); padding:12px; border-radius:var(--radius-md); border:1px solid var(--border-subtle);">
                            <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">PEOPLE OBSERVED</div>
                            <div style="font-size:1.3rem; font-weight:700; color:var(--text-primary);">${data.people_observed}</div>
                        </div>
                        <div style="background:var(--safe-subtle); padding:12px; border-radius:var(--radius-md); border:1px solid var(--safe-border);">
                            <div style="font-size:0.72rem; color:var(--safe-text); font-weight:600;">COMPLIANT</div>
                            <div style="font-size:1.3rem; font-weight:700; color:var(--safe);">${data.compliant_count}</div>
                        </div>
                        <div style="background:var(--warning-subtle); padding:12px; border-radius:var(--radius-md); border:1px solid var(--warning-border);">
                            <div style="font-size:0.72rem; color:var(--warning-text); font-weight:600;">PPE VIOLATIONS</div>
                            <div style="font-size:1.3rem; font-weight:700; color:var(--warning);">${data.violation_count}</div>
                        </div>
                        <div style="background:var(--primary-subtle); padding:12px; border-radius:var(--radius-md); border:1px solid var(--primary-border);">
                            <div style="font-size:0.72rem; color:var(--primary-text); font-weight:600;">OVERALL PPE COMPLIANCE</div>
                            <div style="font-size:1.3rem; font-weight:700; color:var(--primary-text);">${overallCompStr}</div>
                        </div>
                    </div>

                    <!-- Breakdown & Evidence 2-Column Grid -->
                    <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap:24px; margin-bottom:24px; align-items:start;">
                        <!-- Left Column: Violations Breakdown & Person Results -->
                        <div>
                            <!-- PPE Violations Breakdown Strip (Unique Persons) -->
                            <div class="card" style="margin-bottom:18px; border:1px solid var(--border-subtle);">
                                <div class="card-header" style="padding:10px 14px; background:var(--bg-card-subtle);">
                                    <div style="font-weight:700; font-size:0.85rem; color:var(--text-primary);">Unique Person PPE Violations</div>
                                </div>
                                <div class="card-body" style="padding:12px 14px; display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; text-align:center;">
                                    <div style="background:${ppe.hardhat_violations > 0 ? 'var(--critical-subtle)' : 'var(--safe-subtle)'}; padding:8px; border-radius:6px; border:1px solid ${ppe.hardhat_violations > 0 ? 'var(--critical-border)' : 'var(--safe-border)'};">
                                        <div style="font-size:0.72rem; color:var(--text-secondary);">Hardhat Violations</div>
                                        <div style="font-size:1.2rem; font-weight:700; color:${ppe.hardhat_violations > 0 ? 'var(--critical)' : 'var(--safe)'};">${ppe.hardhat_violations || 0}</div>
                                    </div>
                                    <div style="background:${ppe.mask_violations > 0 ? 'var(--critical-subtle)' : 'var(--safe-subtle)'}; padding:8px; border-radius:6px; border:1px solid ${ppe.mask_violations > 0 ? 'var(--critical-border)' : 'var(--safe-border)'};">
                                        <div style="font-size:0.72rem; color:var(--text-secondary);">Mask Violations</div>
                                        <div style="font-size:1.2rem; font-weight:700; color:${ppe.mask_violations > 0 ? 'var(--critical)' : 'var(--safe)'};">${ppe.mask_violations || 0}</div>
                                    </div>
                                    <div style="background:${ppe.vest_violations > 0 ? 'var(--critical-subtle)' : 'var(--safe-subtle)'}; padding:8px; border-radius:6px; border:1px solid ${ppe.vest_violations > 0 ? 'var(--critical-border)' : 'var(--safe-border)'};">
                                        <div style="font-size:0.72rem; color:var(--text-secondary);">Vest Violations</div>
                                        <div style="font-size:1.2rem; font-weight:700; color:${ppe.vest_violations > 0 ? 'var(--critical)' : 'var(--safe)'};">${ppe.vest_violations || 0}</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Person-Wise Cumulative Results -->
                            <div style="font-weight:700; font-size:0.95rem; color:var(--text-primary); margin-bottom:10px;">
                                Person-Wise Result Summary
                            </div>
                            <div style="max-height:360px; overflow-y:auto;">
                                ${personCardsHtml}
                            </div>
                        </div>

                        <!-- Right Column: Representative Evidence Frame & Recommendations -->
                        <div>
                            <div class="card" style="margin-bottom:18px; border:1px solid var(--border-subtle); overflow:hidden;">
                                <div class="card-header" style="padding:10px 14px; background:var(--bg-card-subtle);">
                                    <div style="font-weight:700; font-size:0.85rem; color:var(--text-primary);">
                                        Representative Video Evidence
                                    </div>
                                    <span style="font-size:0.72rem; color:var(--text-muted);">Highest severity frame archived</span>
                                </div>
                                <div style="background:#0B1120; text-align:center; padding:8px;">
                                    <img src="${data.evidence_image_url}?t=${Date.now()}" alt="Video Session Evidence" style="max-width:100%; max-height:260px; object-fit:contain; border-radius:4px; display:block; margin:0 auto;" />
                                </div>
                            </div>

                            <!-- Recommendations -->
                            <div class="card" style="margin-bottom:0; border:1px solid var(--border-subtle);">
                                <div class="card-header" style="padding:10px 14px; background:var(--bg-card-subtle);">
                                    <div style="font-weight:700; font-size:0.85rem; color:var(--text-primary);">Corrective Directives</div>
                                </div>
                                <div class="card-body" style="padding:12px;">
                                    ${recsHtml}
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Bottom Action Bar (Download Report, Session Details, Re-analyze) -->
                    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; padding-top:16px; border-top:1px solid var(--border-subtle);">
                        <a href="${data.report_download_url}" class="btn btn-primary" style="display:inline-flex; align-items:center; gap:8px;">
                            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                            <span>DOWNLOAD VIDEO SAFETY REPORT (PDF)</span>
                        </a>

                        <a href="${data.session_detail_url}" class="btn btn-outline" style="display:inline-flex; align-items:center; gap:6px;">
                            <span>📋 View Audit Details</span>
                        </a>

                        <button type="button" class="btn btn-outline" style="margin-left:auto;" onclick="videoEngine.replayVideo()">
                            <span>↺ Replay & Re-Analyze</span>
                        </button>

                        <button type="button" class="btn btn-outline" onclick="triggerVideoSelect()">
                            <span>Select Another Video</span>
                        </button>
                    </div>

                </div>
            </div>
        `;
    }

    stopAndFinalize() {
        if (!this.videoEl) return;
        this.videoEl.pause();
        this.onVideoEnded();
    }

    replayVideo() {
        if (!this.videoEl) return;
        this.videoEl.currentTime = 0;
        this.isCompleted = false;
        this.lastAnalyzedVideoTime = -1.0;
        if (this.finalResultContainer) this.finalResultContainer.style.display = 'none';
        this.playPause();
    }

    toggleMute() {
        if (!this.videoEl) return;
        this.videoEl.muted = !this.videoEl.muted;
        const btn = document.getElementById('btn-video-volume');
        if (btn) btn.textContent = this.videoEl.muted ? '🔇' : '🔊';
    }

    toggleFullscreen() {
        if (!this.viewportContainer) return;
        if (!document.fullscreenElement) {
            this.viewportContainer.requestFullscreen().catch(err => console.warn(err));
        } else {
            document.exitFullscreen().catch(err => console.warn(err));
        }
    }

    seekVideo(e) {
        if (!this.videoEl || !this.duration) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        this.videoEl.currentTime = pos * this.duration;
    }

    updateScrubber() {
        const bar = document.getElementById('video-progress-bar-fill');
        if (!bar || !this.duration) return;
        const pct = (this.videoEl.currentTime / this.duration) * 100;
        bar.style.width = `${pct}%`;
    }

    updateTimeDisplay() {
        if (!this.timePill) return;
        const cur = this.formatSeconds(this.videoEl ? (this.videoEl.currentTime || 0) : 0);
        const dur = this.duration > 0 ? this.formatSeconds(this.duration) : '--:--';
        this.timePill.textContent = `${cur} / ${dur}`;
    }

    formatSeconds(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    updatePlayPauseButton(isPlaying) {
        const btn = document.getElementById('btn-video-play-pause');
        if (!btn) return;
        if (isPlaying) {
            btn.innerHTML = '<span>⏸ Pause</span>';
            btn.className = 'btn btn-outline btn-sm';
        } else {
            btn.innerHTML = '<span>▶ Play & Analyze</span>';
            btn.className = 'btn btn-primary btn-sm';
        }
    }

    updateStatusPill(text, dotColor) {
        if (!this.statusPill) return;
        this.statusPill.innerHTML = `<span class="pulse-dot" style="background:${dotColor};"></span> ${text}`;
    }

    showVideoError(msg) {
        if (this.errorBanner) {
            this.errorBanner.textContent = `⚠ ${msg}`;
            this.errorBanner.style.display = 'block';
        }
        if (typeof showToast === 'function') {
            showToast('Video Notice', msg, 'WARNING');
        }
    }

    hideVideoError() {
        if (this.errorBanner) {
            this.errorBanner.style.display = 'none';
        }
    }

    updateSessionLabels() {
        const lbl = document.getElementById('session-uid-label-vid');
        if (lbl && this.sessionUid) {
            lbl.textContent = `Session: ${this.sessionUid}`;
        }
    }

    toggleOverlay(propName, btn) {
        if (this.hasOwnProperty(propName)) {
            this[propName] = !this[propName];
            if (btn) {
                if (this[propName]) btn.classList.add('active');
                else btn.classList.remove('active');
            }
            this.redrawOverlay();
        }
    }

    resetUI() {
        if (this.timelineContainer) {
            this.timelineContainer.innerHTML = `
                <div class="timeline-empty-notice" style="text-align:center; padding:16px; color:var(--text-muted); font-size:0.8rem;">
                    Video loaded. Click <b>▶ Play & Analyze</b> to begin real-time PPE detection.
                </div>
            `;
        }

        if (this.currentPeopleContainer) {
            this.currentPeopleContainer.innerHTML = `
                <div style="text-align:center; padding:28px 14px; color:var(--text-muted); background:var(--bg-card-subtle); border-radius:var(--radius-md);">
                    <div style="margin-bottom:6px; font-size:1.4rem;">👤</div>
                    <div style="font-weight:600; font-size:0.85rem; color:var(--text-primary); margin-bottom:2px;">Awaiting Playback</div>
                    <div style="font-size:0.75rem;">Personnel will be analyzed independently as the video plays.</div>
                </div>
            `;
        }

        if (this.finalResultContainer) {
            this.finalResultContainer.style.display = 'none';
            this.finalResultContainer.innerHTML = '';
        }

        const elPeople = document.getElementById('kpi-video-people');
        const elCompliant = document.getElementById('kpi-video-compliant');
        const elViolations = document.getElementById('kpi-video-violations');
        const elCritical = document.getElementById('kpi-video-critical');
        const elComplianceRate = document.getElementById('kpi-video-compliance-rate');
        const elStatusBanner = document.getElementById('video-live-status-banner');

        if (elPeople) elPeople.textContent = '0';
        if (elCompliant) elCompliant.textContent = '0';
        if (elViolations) elViolations.textContent = '0';
        if (elCritical) elCritical.textContent = '0';
        if (elComplianceRate) elComplianceRate.textContent = 'N/A';
        if (elStatusBanner) elStatusBanner.style.display = 'none';
    }
}

// Global initialization & window attachment
let videoEngine = null;

function initVideoDetection() {
    if (window.videoEngine) return;
    videoEngine = new LiveVideoDetectionEngine({
        videoId: 'video-element',
        canvasId: 'video-canvas',
        viewportContainerId: 'video-viewport-wrapper',
        statusPillId: 'video-status-pill',
        timePillId: 'video-time-pill',
        fpsPillId: 'video-fps-pill',
        latencyPillId: 'video-latency-pill',
        timelineContainerId: 'video-event-timeline',
        currentPeopleContainerId: 'video-current-people',
        finalResultContainerId: 'video-final-result-section',
        liveSideSummaryContainerId: 'video-live-summary-card'
    });
    window.videoEngine = videoEngine;
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVideoDetection);
} else {
    initVideoDetection();
}
