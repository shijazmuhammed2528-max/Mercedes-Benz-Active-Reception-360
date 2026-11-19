// Live Capture JavaScript — Mercedes-Benz Active Reception

let statusInterval = null;
let isSessionActive = false;
let currentCameraView = 'all';
let currentCameraMapping = {};
let availableDevicesList = [];

const CAMERAS = ['front', 'back', 'left', 'right'];
const CAM_LABELS = {
    front: 'FRONT VIEW', back: 'REAR VIEW', left: 'LEFT VIEW', right: 'RIGHT VIEW'
};

// ─── Camera Discovery & Screen Assignment ─────────────────

async function loadAvailableCameras(forceRefresh = false) {
    try {
        const res = await fetch(`/api/cameras/available?refresh=${forceRefresh ? 'true' : 'false'}`);
        const data = await res.json();
        availableDevicesList = data.devices || [];
        currentCameraMapping = data.current_mapping || {};

        CAMERAS.forEach(cam => {
            const selectEl = document.getElementById('selectSrc' + capitalize(cam));
            if (!selectEl) return;

            const assignedSource = (currentCameraMapping[cam] && currentCameraMapping[cam].source !== undefined)
                ? String(currentCameraMapping[cam].source)
                : '';

            // Build options
            let optsHtml = '';

            // 1. Hardware Devices
            optsHtml += '<optgroup label="Physical Connected Cameras">';
            availableDevicesList.forEach(dev => {
                const isSelected = (assignedSource === dev.id || assignedSource === String(dev.index));
                const badge = dev.connected ? '● ' : '○ ';
                const statusSuffix = dev.connected ? ' (Connected)' : ' (Not Detected)';
                optsHtml += `<option value="${dev.id}" ${isSelected ? 'selected' : ''}>${badge}${dev.label}${statusSuffix}</option>`;
            });
            optsHtml += '</optgroup>';

            // 2. Demo Video File option
            const isVideoSelected = assignedSource.startsWith('video') || assignedSource.endsWith('.mp4');
            optsHtml += `<optgroup label="Prerecorded File Fallback">
                <option value="video_${cam}" ${isVideoSelected ? 'selected' : ''}>📁 Prerecorded Video (${cam}.mp4)</option>
            </optgroup>`;

            selectEl.innerHTML = optsHtml;

            // Update assigned text label
            updateScreenAssignedBadge(cam, assignedSource);
        });

        // Set initial UI state
        setLiveUIState(Boolean(data.is_live));

    } catch (e) {
        console.error('Failed to load available cameras:', e);
    }
}

function updateScreenAssignedBadge(cam, source) {
    const badgeEl = document.getElementById('assignedLabel' + capitalize(cam));
    if (!badgeEl) return;

    if (!source || source.startsWith('video') || source.endsWith('.mp4')) {
        badgeEl.textContent = `Assigned: Demo File (${cam}.mp4)`;
    } else {
        const dev = availableDevicesList.find(d => d.id === String(source));
        const devName = dev ? dev.label : `Camera ${source}`;
        badgeEl.textContent = `Assigned: ${devName}`;
    }
}

async function assignScreenCamera(screen, source) {
    try {
        const res = await fetch('/api/cameras/assign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ screen, source })
        });
        const data = await res.json();
        if (data.success) {
            updateScreenAssignedBadge(screen, source);
            // If live view is actively running, reload that camera's live MJPEG stream
            if (isSessionActive) {
                const img = document.getElementById('feed' + capitalize(screen));
                if (img) img.src = `/api/capture/stream/${screen}?t=${Date.now()}`;
            }
        } else {
            alert('Failed to assign camera: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        console.error('Assign camera error:', e);
    }
}

async function rescanConnectedCameras() {
    const icon = document.getElementById('scanIcon');
    if (icon) icon.classList.add('fa-spin');
    await loadAvailableCameras(true);
    if (icon) icon.classList.remove('fa-spin');
}

function setLiveUIState(isLive) {
    isSessionActive = isLive;
    const powerInd = document.getElementById('livePowerIndicator');
    const powerText = document.getElementById('powerText');

    if (isLive) {
        if (powerInd) powerInd.classList.add('active');
        if (powerText) powerText.textContent = 'LIVE VIEW ACTIVE (CAMERAS ON)';
        CAMERAS.forEach(cam => {
            const overlay = document.getElementById('overlay' + capitalize(cam));
            if (overlay) overlay.classList.add('hidden');
        });
    } else {
        if (powerInd) powerInd.classList.remove('active');
        if (powerText) powerText.textContent = 'CAMERAS OFF (STANDBY)';
        CAMERAS.forEach(cam => {
            const overlay = document.getElementById('overlay' + capitalize(cam));
            if (overlay) overlay.classList.remove('hidden');
            const statusTxt = document.getElementById('statusText' + capitalize(cam));
            if (statusTxt) {
                statusTxt.textContent = 'CAM OFF';
                statusTxt.style.color = 'var(--text-muted)';
            }
            const dot = document.getElementById('dot' + capitalize(cam));
            if (dot) dot.classList.remove('active');
        });
    }
}

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
    // 1. Load available cameras and assign screen pickers
    await loadAvailableCameras(false);

    // 2. Load capture delay setting
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
            setLiveUIState(true);
            document.getElementById('btnPause').disabled = false;
            document.getElementById('btnStop').disabled = false;

            // Start MJPEG streams
            CAMERAS.forEach(cam => {
                const img = document.getElementById('feed' + capitalize(cam));
                if (img) img.src = `/api/capture/stream/${cam}?t=${Date.now()}`;
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
        stopStatusPolling();
        setLiveUIState(false);

        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnPause').disabled = true;
        document.getElementById('btnResume').disabled = true;
        document.getElementById('btnStop').disabled = true;

        // Clear streams so browser disconnects from camera hardware
        CAMERAS.forEach(cam => {
            const img = document.getElementById('feed' + capitalize(cam));
            if (img) img.src = '';
        });
    } catch (e) {
        console.error('Stop error:', e);
    }
}

async function resetCapture() {
    try {
        await fetch('/api/capture/reset', { method: 'POST' });
        stopStatusPolling();
        setLiveUIState(false);

        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnPause').disabled = true;
        document.getElementById('btnResume').disabled = true;
        document.getElementById('btnStop').disabled = true;
        document.getElementById('captureSuccess').style.display = 'none';

        // Clear streams
        CAMERAS.forEach(cam => {
            const img = document.getElementById('feed' + capitalize(cam));
            if (img) img.src = '';
        });

        // Reset conditions
        resetConditions();
        updateStateIndicator('OFFLINE', 'CAMERAS OFF (STANDBY)');
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

            // Mini status indicator on Camera Selector Bar
            const barDot = document.getElementById('camInd' + capitalize(cam));
            if (barDot) {
                if (c.detected) barDot.classList.add('active');
                else barDot.classList.remove('active');
            }
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

// ─── Camera Selector & View Switching ─────────────────────

function selectCameraView(cam) {
    currentCameraView = cam;
    const grid = document.getElementById('cameraGrid');
    const buttons = {
        all: document.getElementById('btnCamAll'),
        front: document.getElementById('btnCamFront'),
        back: document.getElementById('btnCamBack'),
        left: document.getElementById('btnCamLeft'),
        right: document.getElementById('btnCamRight')
    };

    // Update active tab buttons
    for (const [key, btn] of Object.entries(buttons)) {
        if (btn) {
            if (key === cam) btn.classList.add('active');
            else btn.classList.remove('active');
        }
    }

    const panels = {
        front: document.getElementById('panelFront'),
        back: document.getElementById('panelBack'),
        left: document.getElementById('panelLeft'),
        right: document.getElementById('panelRight')
    };

    if (cam === 'all') {
        if (grid) grid.classList.remove('single-view');
        for (const panel of Object.values(panels)) {
            if (panel) {
                panel.classList.remove('hidden-panel');
                panel.classList.remove('single-focused');
            }
        }
    } else {
        if (grid) grid.classList.add('single-view');
        for (const [key, panel] of Object.entries(panels)) {
            if (panel) {
                if (key === cam) {
                    panel.classList.remove('hidden-panel');
                    panel.classList.add('single-focused');
                } else {
                    panel.classList.add('hidden-panel');
                    panel.classList.remove('single-focused');
                }
            }
        }
    }
}

// ─── Camera Source Mode Switcher ──────────────────────────

async function switchCameraSourceMode(mode) {
    try {
        const res = await fetch('/api/capture/set-source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode })
        });
        const data = await res.json();
        if (data.success) {
            console.log('Switched camera source to:', mode, data.sources);
        } else {
            alert('Source switch failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        console.error('Source switch error:', e);
        alert('Failed to change camera source');
    }
}
