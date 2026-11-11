/**
 * SAFETYVISION AI - Laptop Webcam & Person-Wise PPE Analysis Engine
 * Features:
 * - Direct Laptop Webcam integration with proper browser stream acquisition & error states
 * - Controlled detection cycle (~450ms interval) to keep laptop CPU fast and prevent frame spam
 * - Stable bounding boxes with persistence grace buffer (ZERO blinking)
 * - Independent Person #1, Person #2, Person #3 PPE analysis results
 * - Single representative evidence screenshot captured per live session
 * - Automatic session completion sending to /api/session/live_complete
 * - All fields guarded against undefined/null/NaN
 */

class SafetyVisionMonitor {
    constructor(config) {
        this.videoEl = document.getElementById(config.videoId);
        this.canvasEl = document.getElementById(config.canvasId);
        this.ctx = this.canvasEl ? this.canvasEl.getContext('2d') : null;

        // Overlay Toggles
        this.showBBoxes = true;
        this.showPersonId = true;
        this.showStatusBadge = true;

        this.isRunning = false;
        this.isAnalyzing = false;
        this.isPaused = false;
        this.currentStream = null;
        this.detectionTimer = null;
        this.detectionIntervalMs = 450; // Controlled cycle: ~2.2 AI analyses per second

        // Persistence & Stability Buffer (eliminates blinking on 1-frame drop)
        this.emptyGraceCycles = 0;
        this.maxGraceCycles = 2; // Keep boxes visible for up to 2 empty cycles (~900ms)
        this.lastRenderedAnalysis = null;
        this.lastPersonResults = [];
        this.lastDetections = [];
        this.lastAnalysis = null;

        // Live Session Management
        this.sessionStartTime = null;
        this.capturedEvidence = null;
        this.consolidatedPersons = {}; // key -> person_result

        // Local state-tracking for alert debouncing
        this.personStateCache = {}; // person_id -> status ('COMPLIANT', 'VIOLATION')

        this.initCanvasEvents();
    }

    initCanvasEvents() {
        if (!this.canvasEl) return;
        this.canvasEl.style.pointerEvents = 'auto';
        this.canvasEl.addEventListener('click', (e) => {
            const rect = this.canvasEl.getBoundingClientRect();
            const scaleX = this.canvasEl.width / rect.width;
            const scaleY = this.canvasEl.height / rect.height;
            const clickX = (e.clientX - rect.left) * scaleX;
            const clickY = (e.clientY - rect.top) * scaleY;

            for (let person of this.lastPersonResults) {
                if (!person.bbox) continue;
                const [x1, y1, x2, y2] = person.bbox;
                if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
                    if (typeof openPersonDrawer === 'function') {
                        openPersonDrawer(person);
                    }
                    break;
                }
            }
        });
    }

    async startWebcam() {
        const errorEl = document.getElementById('camera-error-banner');
        if (errorEl) errorEl.style.display = 'none';

        const statusPill = document.getElementById('camera-status-pill');
        if (statusPill) {
            statusPill.innerHTML = '<span class="pulse-dot" style="background:#F59E0B;"></span> CONNECTING...';
        }

        try {
            if (this.currentStream) {
                this.stop();
            }

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("Your browser does not support webcam access. Please use Chrome, Edge, or Firefox.");
            }

            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "user"
                },
                audio: false
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.currentStream = stream;
            this.videoEl.srcObject = stream;
            
            await new Promise((resolve) => {
                this.videoEl.onloadedmetadata = () => {
                    this.videoEl.play();
                    this.canvasEl.width = this.videoEl.videoWidth || 1280;
                    this.canvasEl.height = this.videoEl.videoHeight || 720;
                    resolve();
                };
            });

            this.isRunning = true;
            this.isPaused = false;
            this.isAnalyzing = true;
            this.videoEl.style.visibility = 'visible';
            this.sessionStartTime = Date.now();
            this.capturedEvidence = null;
            this.consolidatedPersons = {};
            this.emptyGraceCycles = 0;
            this.lastRenderedAnalysis = null;

            if (statusPill) {
                statusPill.innerHTML = '<span class="pulse-dot" style="background:#10B981;"></span> CAMERA ACTIVE';
            }
            // Visual only: show the subtle scan line while the camera is genuinely active
            const liveViewport = document.getElementById('video-viewport-wrapper');
            if (liveViewport) liveViewport.classList.add('scanning');

            const btnStart = document.getElementById('btn-stream-start');
            if (btnStart) btnStart.style.display = 'none';
            const btnStop = document.getElementById('btn-stream-stop');
            if (btnStop) btnStop.style.display = 'inline-flex';

            // Hide previous session banner if visible
            const modalEl = document.getElementById('live-session-completed-banner');
            if (modalEl) modalEl.style.display = 'none';

            this.startDetectionCycle();
            if (typeof showToast === 'function') {
                showToast('Laptop Webcam Active', 'Live video stream connected. Person PPE Analysis running.', 'INFO');
            }
        } catch (err) {
            console.error('Camera connection error:', err);
            this.showCameraError(err);
        }
    }

    showCameraError(err) {
        let msg = "Could not access laptop webcam.";
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            msg = "Camera permission was blocked. Please click the camera icon in your browser address bar, allow camera access, and reload.";
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            msg = "No webcam detected on this laptop. Please check your camera hardware connection.";
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            msg = "Webcam is already in use by another application. Please close other camera apps and retry.";
        } else {
            msg = err.message || msg;
        }

        const errorEl = document.getElementById('camera-error-banner');
        if (errorEl) {
            errorEl.textContent = msg;
            errorEl.style.display = 'block';
        }

        const statusPill = document.getElementById('camera-status-pill');
        if (statusPill) {
            statusPill.innerHTML = '<span style="color:#EF4444;">● ERROR</span>';
        }
        // Visual only: no scan line in the error state
        const errViewport = document.getElementById('video-viewport-wrapper');
        if (errViewport) errViewport.classList.remove('scanning');

        if (typeof showToast === 'function') {
            showToast('Camera Error', msg, 'CRITICAL');
        }
    }

    async stop() {
        this.isRunning = false;
        this.isAnalyzing = false;

        if (this.detectionTimer) {
            clearInterval(this.detectionTimer);
            this.detectionTimer = null;
        }

        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }

        if (this.videoEl) {
            this.videoEl.pause();
            this.videoEl.srcObject = null;
            this.videoEl.style.visibility = 'hidden';
        }

        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
        }

        const btnStart = document.getElementById('btn-stream-start');
        if (btnStart) btnStart.style.display = 'inline-flex';
        const btnStop = document.getElementById('btn-stream-stop');
        if (btnStop) btnStop.style.display = 'none';

        const statusPill = document.getElementById('camera-status-pill');
        if (statusPill) {
            statusPill.innerHTML = '<span style="color:#94A3B8;">● READY</span>';
        }
        // Visual only: hide the scan line once the camera stops
        const stopViewport = document.getElementById('video-viewport-wrapper');
        if (stopViewport) stopViewport.classList.remove('scanning');

        // Finalize Live Session if we detected people
        const personsList = Object.values(this.consolidatedPersons);
        const durationSec = this.sessionStartTime ? Math.round((Date.now() - this.sessionStartTime) / 1000) : 0;

        if (personsList.length > 0 || this.capturedEvidence) {
            try {
                const response = await fetch('/api/session/live_complete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content },
                    body: JSON.stringify({
                        image_data: this.capturedEvidence || '',
                        person_results: personsList,
                        duration_seconds: durationSec
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    if (data.session_uid) {
                        this.showLiveSessionCompletedNotice(data);
                    }
                }
            } catch (e) {
                console.warn('Error saving live session:', e);
            }
        }

        this.personStateCache = {};
    }

    showLiveSessionCompletedNotice(sessionData) {
        let banner = document.getElementById('live-session-completed-banner');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'live-session-completed-banner';
            const wrapper = document.getElementById('video-viewport-wrapper');
            if (wrapper && wrapper.parentNode) {
                wrapper.parentNode.insertBefore(banner, wrapper);
            }
        }

        if (banner) {
            banner.style.cssText = 'background:var(--primary-subtle); border:1px solid var(--primary-border); border-radius:8px; padding:16px 20px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;';
            banner.innerHTML = `
                <div>
                    <div style="font-size:0.75rem; font-weight:700; color:var(--primary-text); text-transform:uppercase;">Live Session Recorded</div>
                    <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary); margin-top:2px;">
                        Session <span style="font-family:monospace; color:var(--primary-text);">${sessionData.session_uid}</span>
                    </div>
                    <div style="font-size:0.82rem; color:var(--text-secondary); margin-top:4px;">
                        ${sessionData.people_count || 0} Persons Analyzed &bull; ${sessionData.overall_compliance || 100}% Compliance &bull; 1 Evidence Screenshot Archived
                    </div>
                </div>
                <div style="display:flex; gap:10px;">
                    <a href="/session/${sessionData.session_uid}" class="btn btn-outline btn-sm">View Session</a>
                    <a href="/reports/session/${sessionData.session_uid}/download" class="btn btn-primary btn-sm" style="display:inline-flex; align-items:center; gap:6px;">
                        <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        Download Session Report (PDF)
                    </a>
                </div>
            `;
            banner.style.display = 'flex';
        }
    }

    pause() {
        this.isPaused = !this.isPaused;
        return this.isPaused;
    }

    startDetectionCycle() {
        const offCanvas = document.createElement('canvas');
        const offCtx = offCanvas.getContext('2d');
        let isSending = false;

        this.detectionTimer = setInterval(async () => {
            if (!this.isRunning || this.isPaused || isSending || !this.isAnalyzing) return;
            if (this.videoEl.readyState < 2) return;

            const vw = this.videoEl.videoWidth;
            const vh = this.videoEl.videoHeight;
            if (vw === 0 || vh === 0) return;

            this.canvasEl.width = vw;
            this.canvasEl.height = vh;
            offCanvas.width = vw;
            offCanvas.height = vh;

            offCtx.drawImage(this.videoEl, 0, 0, vw, vh);
            const dataUrl = offCanvas.toDataURL('image/jpeg', 0.70);

            isSending = true;
            try {
                const response = await fetch('/api/detect/frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content },
                    body: JSON.stringify({ image_data: dataUrl, auto_persist: true })
                });

                if (response.ok) {
                    const result = await response.json();
                    this.lastAnalysis = result;
                    const curPersons = result.person_results || result.tracked_persons || [];
                    this.lastDetections = result.raw_detections || [];

                    // Update consolidated records for session summary
                    curPersons.forEach(p => {
                        const key = p.label || `Person #${p.id || 1}`;
                        this.consolidatedPersons[key] = p;
                    });

                    // Capture ONE primary evidence screenshot once people are confirmed
                    if (!this.capturedEvidence && curPersons.length > 0) {
                        const snapCanvas = document.createElement('canvas');
                        snapCanvas.width = vw;
                        snapCanvas.height = vh;
                        const snapCtx = snapCanvas.getContext('2d');
                        snapCtx.drawImage(this.videoEl, 0, 0, vw, vh);
                        this.capturedEvidence = snapCanvas.toDataURL('image/jpeg', 0.85);
                    }

                    // 1. Render Video Bounding Boxes with Stability
                    this.renderOverlay(result);

                    // 2. Render Individual Person-Wise Cards
                    this.renderPersonResults(this.lastPersonResults);

                    // 3. Update Summary KPI Cards
                    this.updateSummaryKPIs(result);

                    // 4. Handle State-Based Alerts (NO FRAME SPAM)
                    if (result.new_alerts && result.new_alerts.length > 0) {
                        result.new_alerts.forEach(alert => {
                            if (typeof showToast === 'function') {
                                showToast(alert.title, alert.message, alert.severity);
                            }
                            this.appendTimelineEvent(alert);
                        });
                    }
                }
            } catch (err) {
                console.warn('Analysis cycle error:', err);
            } finally {
                isSending = false;
            }
        }, this.detectionIntervalMs);
    }

    renderOverlay(analysis) {
        if (!this.ctx) return;

        let persons = analysis.person_results || analysis.tracked_persons || [];
        let rawDets = analysis.raw_detections || [];

        // Persistence logic: if detection dropped for 1 cycle, keep last stable render
        if (persons.length === 0) {
            if (this.emptyGraceCycles < this.maxGraceCycles && this.lastRenderedAnalysis) {
                persons = this.lastRenderedAnalysis.person_results || this.lastRenderedAnalysis.tracked_persons || [];
                rawDets = this.lastRenderedAnalysis.raw_detections || [];
                this.emptyGraceCycles++;
            } else {
                this.emptyGraceCycles = 0;
                this.lastRenderedAnalysis = null;
            }
        } else {
            this.emptyGraceCycles = 0;
            this.lastRenderedAnalysis = analysis;
        }

        this.lastPersonResults = persons;

        // Clean redraw
        this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);

        if (!this.showBBoxes) return;

        // Track drawn PPE boxes to avoid duplicate drawing
        const drawnPPEBoxes = new Set();

        // 1. Draw Environmental Hazards (Machinery, Vehicles, Cones)
        rawDets.forEach(det => {
            const cname = det.class_name;
            if (!cname || cname === 'Person' || cname.includes('Hardhat') || cname.includes('Mask') || cname.includes('Vest') || !det.bbox) return;
            const [x1, y1, x2, y2] = det.bbox;
            const conf = det.confidence ? Math.round(det.confidence * 100) : 80;
            const color = det.color || '#F59E0B';
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            const tag = `${cname} ${conf}%`;
            this.ctx.font = 'bold 11px Inter, sans-serif';
            const tw = this.ctx.measureText(tag).width;
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.88)';
            this.ctx.fillRect(x1, Math.max(0, y1 - 18), tw + 8, 18);
            this.ctx.fillStyle = '#111827';
            this.ctx.fillText(tag, x1 + 4, Math.max(12, y1 - 4));
        });

        // 2. Draw Detected PPE Bounding Boxes (with explicit class & confidence)
        persons.forEach(person => {
            const items = person.associated_ppe || [];
            items.forEach(it => {
                if (!it.bbox) return;
                const [ix1, iy1, ix2, iy2] = it.bbox;
                const boxKey = `${Math.round(ix1)}_${Math.round(iy1)}_${Math.round(ix2)}_${Math.round(iy2)}`;
                drawnPPEBoxes.add(boxKey);

                const cname = it.class_name || 'PPE';
                const conf = it.confidence ? Math.round(it.confidence * 100) : 85;
                const isNeg = cname.startsWith('NO-');
                const boxColor = isNeg ? '#EF4444' : (cname.includes('Mask') ? '#06B6D4' : (cname.includes('Hardhat') ? '#10B981' : '#3B82F6'));

                this.ctx.strokeStyle = boxColor;
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(ix1, iy1, ix2 - ix1, iy2 - iy1);

                const tag = `${cname} ${conf}%`;
                this.ctx.font = 'bold 10px Inter, sans-serif';
                const tw = this.ctx.measureText(tag).width;
                const tagY = Math.min(this.canvasEl.height - 4, iy2 + 14);
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.88)';
                this.ctx.fillRect(ix1, tagY - 12, tw + 8, 14);
                this.ctx.fillStyle = '#111827';
                this.ctx.fillText(tag, ix1 + 4, tagY - 2);
            });
        });

        // Draw any unassociated PPE in rawDets
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
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(ix1, iy1, ix2 - ix1, iy2 - iy1);

            const tag = `${cname} ${conf}%`;
            this.ctx.font = 'bold 10px Inter, sans-serif';
            const tw = this.ctx.measureText(tag).width;
            const tagY = Math.min(this.canvasEl.height - 4, iy2 + 14);
            this.ctx.fillStyle = boxColor;
            this.ctx.fillRect(ix1, tagY - 12, tw + 8, 14);
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.fillText(tag, ix1 + 4, tagY - 2);
        });

        // 3. Draw Person Outer Bounding Boxes & Primary Status Labels
        persons.forEach(person => {
            if (!person.bbox) return;
            const [x1, y1, x2, y2] = person.bbox;
            const w = x2 - x1;
            const h = y2 - y1;

            const comp = (person.compliance !== undefined && !isNaN(person.compliance)) ? person.compliance : 100;
            const isCompliant = comp >= 99.0;
            const risk = person.risk || (isCompliant ? 'LOW' : 'MEDIUM');

            let color = '#10B981';
            if (risk === 'CRITICAL') color = '#EF4444';
            else if (risk === 'HIGH') color = '#F97316';
            else if (risk === 'MEDIUM') color = '#F59E0B';

            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x1, y1, w, h);

            if (this.showPersonId) {
                const label = person.label || `Person #${person.id || 1}`;
                const statusText = person.status || (isCompliant ? 'COMPLIANT' : 'VIOLATION');
                const tag = `${label} | PPE: ${comp}% | ${statusText}`;
                
                this.ctx.font = 'bold 12px Inter, sans-serif';
                const tagW = this.ctx.measureText(tag).width;

                this.ctx.fillStyle = color;
                this.ctx.fillRect(x1, Math.max(0, y1 - 24), tagW + 14, 24);

                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.fillText(tag, x1 + 6, Math.max(16, y1 - 7));
            }

            if (this.showStatusBadge && person.missing_items && person.missing_items.length > 0) {
                const missingText = `Missing: ${person.missing_items.join(', ')}`;
                this.ctx.font = '11px Inter, sans-serif';
                const mw = this.ctx.measureText(missingText).width;

                this.ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
                this.ctx.fillRect(x1, y2, mw + 14, 20);
                this.ctx.fillStyle = '#FCA5A5';
                this.ctx.fillText(missingText, x1 + 7, y2 + 14);
            }
        });
    }

    renderPersonResults(personResults) {
        const container = document.getElementById('person-results-container');
        if (!container) return;

        if (!personResults || personResults.length === 0) {
            container.innerHTML = `
                <div style="text-align:center; padding:32px 16px; color:var(--text-muted); background:var(--bg-card-subtle); border-radius:var(--radius-md);">
                    <div style="font-size:1.8rem; margin-bottom:6px;">👤</div>
                    <div style="font-weight:600; font-size:0.85rem; color:var(--text-primary); margin-bottom:2px;">No Personnel in Camera View</div>
                    <div style="font-size:0.75rem;">Stand in front of the laptop webcam to analyze PPE compliance.</div>
                </div>
            `;
            return;
        }

        let html = '';
        personResults.forEach(p => {
            const label = p.label || `Person #${p.id || 1}`;
            const comp = (p.compliance !== undefined && !isNaN(p.compliance)) ? p.compliance : 100;
            const isCompliant = comp >= 99.0;
            const status = p.status || (isCompliant ? 'COMPLIANT' : 'PPE VIOLATION');
            const risk = p.risk || (isCompliant ? 'LOW' : 'MEDIUM');
            const duration = p.duration_seconds !== undefined ? Math.round(p.duration_seconds) : 0;

            const statusClass = isCompliant ? 'badge-low' : (risk === 'CRITICAL' ? 'badge-critical' : 'badge-high');
            const borderColor = isCompliant ? '#10B981' : (risk === 'CRITICAL' ? '#EF4444' : '#F59E0B');

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
                <div class="card" style="margin-bottom:12px; border-left:4px solid ${borderColor}; box-shadow:var(--shadow-sm);">
                    <div style="padding:12px 14px; background:var(--bg-card-subtle); display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-subtle);">
                        <div>
                            <div style="font-family:var(--font-display); font-weight:700; font-size:0.95rem; color:var(--text-primary);">${label}</div>
                            <div style="font-size:0.72rem; color:var(--text-muted);">Duration: ${duration}s</div>
                        </div>
                        <span class="badge ${statusClass}">${status}</span>
                    </div>
                    <div style="padding:12px 14px; background:var(--bg-card);">
                        ${renderGearRow('Hardhat', p.hardhat || 'MISSING', p.hardhat_label, p.hardhat_conf)}
                        ${renderGearRow('Protective Mask', p.mask || 'MISSING', p.mask_label, p.mask_conf)}
                        ${renderGearRow('Safety Vest', p.vest || 'MISSING', p.vest_label, p.vest_conf)}

                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border-subtle); font-size:0.78rem;">
                            <span>Compliance: <b style="color:${isCompliant ? '#10B981' : '#DC2626'}; font-size:0.9rem;">${comp}%</b></span>
                            <span>Risk: <b style="color:${borderColor};">${risk}</b></span>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    }

    updateSummaryKPIs(analysis) {
        const m = analysis.live_metrics || {};
        const persons = analysis.person_results || analysis.tracked_persons || [];

        const totalPersons = m.total_persons !== undefined ? m.total_persons : persons.length;
        const compliantPersons = m.compliant_persons !== undefined ? m.compliant_persons : 0;
        const violatingPersons = m.violating_persons !== undefined ? m.violating_persons : 0;
        const criticalCount = m.critical_count !== undefined ? m.critical_count : 0;
        const complianceRate = (m.compliance_rate !== undefined && m.compliance_rate !== 'N/A' && !isNaN(m.compliance_rate)) ? m.compliance_rate : null;

        // HUD Pills
        const fpsEl = document.getElementById('hud-fps');
        if (fpsEl) fpsEl.textContent = `${analysis.fps || 0} FPS`;

        const latEl = document.getElementById('hud-latency');
        if (latEl) latEl.textContent = `${analysis.latency_ms || 0} ms`;

        // KPI Numbers
        const pTotalEl = document.getElementById('kpi-people-detected');
        if (pTotalEl) pTotalEl.textContent = totalPersons;

        const pCompEl = document.getElementById('kpi-compliant-count');
        if (pCompEl) pCompEl.textContent = compliantPersons;

        const pViolEl = document.getElementById('kpi-violations-count');
        // Canonical: PPE Violations = individually missing items (not people).
        const itemViol = (m.ppe_violations !== undefined && m.ppe_violations !== null)
            ? m.ppe_violations
            : persons.reduce((acc, p) => acc
                + ((p && p.hardhat === 'MISSING') ? 1 : 0)
                + ((p && p.mask === 'MISSING') ? 1 : 0)
                + ((p && p.vest === 'MISSING') ? 1 : 0), 0);
        if (pViolEl) { pViolEl.textContent = itemViol; pViolEl.title = 'Individually missing PPE items'; }

        const pCritEl = document.getElementById('kpi-critical-count');
        if (pCritEl) pCritEl.textContent = criticalCount;

        const pRateEl = document.getElementById('kpi-compliance-rate');
        if (pRateEl) {
            pRateEl.textContent = (totalPersons > 0 && complianceRate !== null) ? `${complianceRate}%` : 'N/A';
        }
    }

    appendTimelineEvent(alert) {
        const timelineEl = document.getElementById('event-timeline');
        if (!timelineEl) return;

        const timeStr = new Date().toLocaleTimeString();
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.style.borderLeftColor = alert.severity === 'CRITICAL' ? 'var(--critical)' : (alert.type === 'COMPLIANCE_RESTORED' ? 'var(--safe)' : 'var(--warning)');

        item.innerHTML = `
            <div class="timeline-time">${timeStr}</div>
            <div class="timeline-content">
                <b>${alert.title || 'Safety Event'}</b> &bull; ${alert.message || ''}
            </div>
        `;

        timelineEl.prepend(item);
        while (timelineEl.children.length > 25) {
            timelineEl.removeChild(timelineEl.lastChild);
        }
    }

    captureSnapshot() {
        if (!this.canvasEl) return;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvasEl.width;
        tempCanvas.height = this.canvasEl.height;
        const tempCtx = tempCanvas.getContext('2d');

        if (this.videoEl && this.videoEl.videoWidth > 0) {
            tempCtx.drawImage(this.videoEl, 0, 0, tempCanvas.width, tempCanvas.height);
        }
        tempCtx.drawImage(this.canvasEl, 0, 0);

        const link = document.createElement('a');
        link.download = `SafetyVision_Snapshot_${Date.now()}.png`;
        link.href = tempCanvas.toDataURL('image/png');
        link.click();
        if (typeof showToast === 'function') {
            showToast('Snapshot Saved', 'Webcam frame saved to your downloads', 'INFO');
        }
    }

    toggleFullscreen() {
        const wrapper = document.getElementById('video-viewport-wrapper');
        if (!document.fullscreenElement) {
            wrapper.requestFullscreen().catch(err => alert(err.message));
        } else {
            document.exitFullscreen();
        }
    }
}
