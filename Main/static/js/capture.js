// Live Capture JavaScript — Mercedes-Benz Active Reception

let statusInterval = null;
let isSessionActive = false;

const CAMERAS = ['front', 'back', 'left', 'right'];
const CAM_LABELS = {
    front: 'FRONT', back: 'REAR', left: 'LEFT', right: 'RIGHT'
};

// ─── Capture Controls ──────────────────────────────────────

async function changeCaptureDelay(val) {
    const delay = parseFloat(val);
    try {
        const res = await fetch('/api/capture/set-delay', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ delay })
        });
        const data = await res.json();
        console.log('Capture delay updated:', data.capture_delay_seconds);
    } catch (e) {
        console.error('Failed to update capture delay:', e);
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/settings');
        const cfg = await res.json();
        if (cfg.capture_delay_seconds !== undefined) {
            const sel = document.getElementById('selectCaptureDelay');
            if (sel) sel.value = parseFloat(cfg.capture_delay_seconds).toFixed(1);
        }
    } catch (e) {}
});

async function startCapture() {
    try {
        document.getElementById('btnStart').disabled = true;
        document.getElementById('captureSuccess').style.display = 'none';

        const res = await fetch('/api/capture/start', { method: 'POST' });
        const data = await res.json();

        if (data.success) {
            isSessionActive = true;
            document.getElementById('btnPause').disabled = false;
            document.getElementById('btnStop').disabled = false;

            // Start MJPEG streams
            CAMERAS.forEach(cam => {
                const img = document.getElementById('feed' + capitalize(cam));
                img.src = `/api/capture/stream/${cam}?t=${Date.now()}`;
            });

            // Start status polling
            startStatusPolling();
        } else {
            alert('Start failed: ' + (data.error || 'Unknown error'));
            document.getElementById('btnStart').disabled = false;
        }
    } catch (e) {
        console.error('Start error:', e);
        document.getElementById('btnStart').disabled = false;
        alert('Failed to start capture session');
    }
}

async function pauseCapture() {
    try {
        await fetch('/api/capture/pause', { method: 'POST' });
        document.getElementById('btnPause').disabled = true;
        document.getElementById('btnResume').disabled = false;
    } catch (e) {
        console.error('Pause error:', e);
    }
}

async function resumeCapture() {
    try {
        await fetch('/api/capture/resume', { method: 'POST' });
        document.getElementById('btnPause').disabled = false;
        document.getElementById('btnResume').disabled = true;
    } catch (e) {
        console.error('Resume error:', e);
    }
}

async function stopCapture() {
    try {
        await fetch('/api/capture/stop', { method: 'POST' });
        isSessionActive = false;
        stopStatusPolling();

        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnPause').disabled = true;
        document.getElementById('btnResume').disabled = true;
        document.getElementById('btnStop').disabled = true;

        // Clear streams
        CAMERAS.forEach(cam => {
            document.getElementById('feed' + capitalize(cam)).src = '';
        });
    } catch (e) {
        console.error('Stop error:', e);
    }
}

async function resetCapture() {
    try {
        await fetch('/api/capture/reset', { method: 'POST' });
        isSessionActive = false;
        stopStatusPolling();

        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnPause').disabled = true;
        document.getElementById('btnResume').disabled = true;
        document.getElementById('btnStop').disabled = true;
        document.getElementById('captureSuccess').style.display = 'none';

        // Clear streams
        CAMERAS.forEach(cam => {
            document.getElementById('feed' + capitalize(cam)).src = '';
        });

        // Reset conditions
        resetConditions();
        updateStateIndicator('INITIALIZING', 'SYSTEM READY');
    } catch (e) {
        console.error('Reset error:', e);
    }
}

// ─── Plate Correction ──────────────────────────────────────

async function savePlate() {
    const plate = document.getElementById('plateInput').value.trim();
    if (!plate) return;

    try {
        const res = await fetch('/api/capture/set-plate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plate })
        });
        const data = await res.json();
        if (data.success) {
            document.getElementById('successPlate').textContent = data.plate;
        }
    } catch (e) {
        console.error('Set plate error:', e);
    }
}

// ─── Status Polling ────────────────────────────────────────

function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(pollStatus, 500);
}

function stopStatusPolling() {
    if (statusInterval) {
        clearInterval(statusInterval);
        statusInterval = null;
    }
}

async function pollStatus() {
    try {
        const res = await fetch('/api/capture/status');
        const data = await res.json();

        // Update state indicator
        const cs = data.capture_state || {};
        updateStateIndicator(cs.state, cs.message);

        // Update camera conditions
        const conds = data.conditions || {};
        for (const cam of CAMERAS) {
            const c = conds[cam] || {};

            // Detection dot
            const dot = document.getElementById('dot' + capitalize(cam));
            const statusText = document.getElementById('statusText' + capitalize(cam));
            if (c.detected) {
                dot.classList.add('active');
                statusText.textContent = 'TRACKING';
                statusText.style.color = 'var(--success)';
            } else {
                dot.classList.remove('active');
                statusText.textContent = 'SEARCHING';
                statusText.style.color = 'var(--text-muted)';
            }

            // Confidence
            const conf = document.getElementById('conf' + capitalize(cam));
            conf.textContent = c.detected ? `${(c.confidence * 100).toFixed(1)}%` : '—';

            // Motion
            const motion = document.getElementById('motion' + capitalize(cam));
            motion.textContent = c.motion_state || '—';

            // Conditions checklist
            updateCondition(`cond${capitalize(cam)}Detected`, c.detected);
            updateCondition(`cond${capitalize(cam)}View`, c.view_ok);
        }

        // Stationary condition (all cameras)
        const allStationary = CAMERAS.every(cam => (conds[cam] || {}).stationary);
        updateCondition('condStationary', allStationary);

        // Delay condition / countdown
        const delayLabel = document.getElementById('delayCountdownLabel');
        if (data.ready_countdown !== null && data.ready_countdown !== undefined) {
            updateCondition('condDelayReady', true);
            if (delayLabel) {
                delayLabel.textContent = `Capturing in ${data.ready_countdown.toFixed(1)}s...`;
            }
        } else if (data.captured) {
            updateCondition('condDelayReady', true);
            if (delayLabel) delayLabel.textContent = 'Capture complete';
        } else {
            updateCondition('condDelayReady', false);
            if (delayLabel) delayLabel.textContent = 'Capture delay verified';
        }

        // Capture success
        if (data.captured && data.capture_result) {
            showCaptureSuccess(data.capture_result, data.plate_text);
        }

    } catch (e) {
        // Silent — will retry next interval
    }
}

// ─── UI Updates ────────────────────────────────────────────

function updateStateIndicator(state, message) {
    const dot = document.getElementById('stateDot');
    const text = document.getElementById('stateText');
    text.textContent = message || state;

    // Remove all state classes
    dot.className = 'state-dot';

    switch (state) {
        case 'MONITORING':
            dot.classList.add('monitoring'); break;
        case 'VEHICLE_DETECTED':
        case 'VEHICLE_MOVING':
        case 'VEHICLE_STOPPING':
        case 'CHECKING_VIEWS':
            dot.classList.add('detected'); break;
        case 'CAPTURE_READY':
        case 'CAPTURING':
            dot.classList.add('capturing'); break;
        case 'CAPTURED':
        case 'SAVED':
            dot.classList.add('saved'); break;
        case 'ERROR':
            dot.classList.add('error'); break;
    }
}

function updateCondition(id, passed) {
    const el = document.getElementById(id);
    if (!el) return;
    if (passed) {
        el.classList.add('passed');
    } else {
        el.classList.remove('passed');
    }
}

function resetConditions() {
    document.querySelectorAll('.condition-item').forEach(el => {
        el.classList.remove('passed');
    });
}

function showCaptureSuccess(result, plateText) {
    const el = document.getElementById('captureSuccess');
    el.style.display = 'block';

    const plate = plateText || result.vehicle_number || '';
    const isUnknown = (!plate || plate === 'UNKNOWN');
    const plateDisplay = isUnknown ? 'REGISTRATION NOT CONFIRMED' : plate;
    const plateVal = isUnknown ? '' : plate;

    const plateEl = document.getElementById('successPlate');
    plateEl.textContent = plateDisplay;
    if (isUnknown) {
        plateEl.style.color = 'var(--warning)';
        plateEl.style.fontSize = '20px';
    } else {
        plateEl.style.color = 'var(--text-primary)';
        plateEl.style.fontSize = '28px';
    }

    document.getElementById('successMeta').textContent =
        `${result.capture_date} — ${result.capture_time_12h}`;
    const inputEl = document.getElementById('plateInput');
    inputEl.value = plateVal;
    if (isUnknown) {
        inputEl.placeholder = 'ENTER REGISTRATION (e.g. KL 01 AL 0009)';
    }

    // Preview images
    const preview = document.getElementById('capturePreview');
    preview.innerHTML = CAMERAS.map(cam =>
        `<img src="/api/vehicle/thumbnail/${result.session_id}/${cam}"
              onerror="this.style.background='var(--bg-surface-3)'" alt="${cam}">`
    ).join('');

    // Links
    document.getElementById('linkViewSession').href = `/vehicle/${result.session_id}`;
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

// ─── Video Test & Diagnostic Modal ─────────────────────────

async function openLiveVideoTestModal() {
    const modal = document.getElementById('liveVideoTestModal');
    const content = document.getElementById('liveVideoTestContent');
    modal.style.display = 'flex';
    content.innerHTML = '<div style="padding:40px; text-align:center; color:var(--text-muted);"><i class="fas fa-spinner fa-spin fa-2x"></i><div style="margin-top:14px; font-size:14px;">Decoding frames and running YOLO vehicle detection across all 4 cameras...</div></div>';

    try {
        const res = await fetch('/api/videos/test');
        const data = await res.json();
        const cams = data.cameras || {};

        let cardsHtml = '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px;">';

        for (const [cam, info] of Object.entries(cams)) {
            const isOk = info.status === 'PASSED';
            const badgeColor = isOk ? 'var(--success)' : 'var(--error)';
            const detHtml = info.vehicle_detected 
                ? `<span style="color:var(--success); font-weight:700;"><i class="fas fa-check-circle"></i> ${info.detection_class.toUpperCase()} (${info.detection_confidence}%)</span>`
                : `<span style="color:var(--warning);"><i class="fas fa-exclamation-circle"></i> No vehicle detected</span>`;

            cardsHtml += `
                <div style="background:#181818; border:1px solid ${isOk ? '#2a2a2a' : '#552222'}; border-radius:6px; padding:14px; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-weight:700; font-size:13px; letter-spacing:0.05em; color:#fff;">CAM ${cam.toUpperCase()} (${cam}.mp4)</span>
                            <span style="font-size:10px; font-weight:700; padding:2px 8px; border-radius:3px; background:${isOk ? 'rgba(0,255,102,0.1)' : 'rgba(255,50,50,0.1)'}; color:${badgeColor}; border:1px solid ${badgeColor};">
                                ${info.status}
                            </span>
                        </div>
                        ${info.preview_jpeg_b64 ? `
                            <div style="position:relative; width:100%; height:130px; background:#000; border-radius:4px; overflow:hidden; margin-bottom:10px;">
                                <img src="data:image/jpeg;base64,${info.preview_jpeg_b64}" style="width:100%; height:100%; object-fit:cover;" alt="${cam}">
                            </div>
                        ` : ''}
                        <div style="font-size:11px; color:#aaa; line-height:1.7;">
                            <div><strong>Resolution:</strong> ${info.width}x${info.height}</div>
                            <div><strong>Frame Rate:</strong> ${info.fps} FPS (${info.frame_count} frames)</div>
                            <div><strong>Duration:</strong> ${info.duration_seconds}s</div>
                            <div><strong>YOLO Check:</strong> ${detHtml}</div>
                        </div>
                    </div>
                    <div style="margin-top:12px;">
                        <button class="btn btn-sm" onclick="playLiveVideo('${cam}')" style="width:100%; justify-content:center; font-size:11px;">
                            <i class="fas fa-play"></i> PLAY ${cam.toUpperCase()}.MP4
                        </button>
                    </div>
                </div>
            `;
        }
        cardsHtml += '</div>';

        content.innerHTML = `
            <div style="padding:14px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <strong style="font-size:13px; color:${data.all_passed ? 'var(--success)' : 'var(--error)'};">
                        <i class="fas fa-${data.all_passed ? 'check-circle' : 'exclamation-triangle'}"></i> 
                        ${data.all_passed ? 'ALL 4 PRERECORDED VIDEOS PASSED DIAGNOSTICS & DETECTION' : 'ONE OR MORE VIDEOS FAILED'}
                    </strong>
                    <button class="btn btn-sm btn-primary" onclick="openLiveVideoTestModal()" style="padding:4px 10px; font-size:11px;"><i class="fas fa-redo"></i> RE-TEST</button>
                </div>
                ${cardsHtml}
            </div>
        `;
    } catch (e) {
        console.error('Video test error:', e);
        content.innerHTML = '<div style="color:var(--error); padding:20px;">Diagnostic test failed: ' + e.message + '</div>';
    }
}

function closeLiveVideoTestModal() {
    const modal = document.getElementById('liveVideoTestModal');
    if (modal) modal.style.display = 'none';
}

function playLiveVideo(camera) {
    const modal = document.getElementById('livePlayerModal');
    const player = document.getElementById('liveVideoTag');
    const title = document.getElementById('livePlayerTitle');

    title.textContent = `PRERECORDED VIDEO: CAM ${camera.toUpperCase()} (${camera}.mp4)`;
    player.src = `/api/videos/raw/${camera}?t=${Date.now()}`;
    modal.style.display = 'flex';
    player.play().catch(e => console.log('Autoplay prevented:', e));
}

function closeLiveVideoPlayer() {
    const modal = document.getElementById('livePlayerModal');
    const player = document.getElementById('liveVideoTag');
    player.pause();
    player.src = '';
    modal.style.display = 'none';
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeLiveVideoTestModal();
        closeLiveVideoPlayer();
    }
});
