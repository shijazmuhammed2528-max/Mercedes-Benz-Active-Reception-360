// Live Capture JavaScript — Mercedes-Benz Active Reception

let statusInterval = null;
let isSessionActive = false;

const CAMERAS = ['front', 'back', 'left', 'right'];
const CAM_LABELS = {
    front: 'FRONT', back: 'REAR', left: 'LEFT', right: 'RIGHT'
};

// ─── Capture Controls ──────────────────────────────────────

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

    document.getElementById('successPlate').textContent = plateText || result.vehicle_number || 'UNKNOWN';
    document.getElementById('successMeta').textContent =
        `${result.capture_date} — ${result.capture_time_12h}`;
    document.getElementById('plateInput').value = plateText || result.vehicle_number || '';

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
