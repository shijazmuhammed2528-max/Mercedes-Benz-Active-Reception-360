// Live Capture JavaScript — Mercedes-Benz Active Reception

let statusInterval = null;
let isSessionActive = false;
let isCameraPowerOn = true;
let isAutoCaptureActive = true;
let currentCameraView = 'all';
let currentCameraMapping = {};
let availableDevicesList = [];
let wifiCamerasList = [];
let pendingWifiScreenTarget = null;

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
        wifiCamerasList = data.wifi_cameras || [];
        currentCameraMapping = data.current_mapping || {};

        CAMERAS.forEach(cam => {
            const selectEl = document.getElementById('selectSrc' + capitalize(cam));
            if (!selectEl) return;

            // If user has focused this dropdown, don't clobber it while they are selecting
            if (document.activeElement === selectEl) return;

            const assignedSource = (currentCameraMapping[cam] && currentCameraMapping[cam].source !== undefined)
                ? String(currentCameraMapping[cam].source)
                : '';

            // Group 1: Hardware USB / Integrated cameras (Connected first)
            let optsHtml = '<optgroup label="Physical USB & Webcams (Select Any)">';
            const sortedDevs = [...availableDevicesList].sort((a, b) => (b.connected ? 1 : 0) - (a.connected ? 1 : 0));
            sortedDevs.forEach(dev => {
                const isSelected = (assignedSource === dev.id || assignedSource === String(dev.index));
                const badge = dev.connected ? '🟢 ' : '🔴 ';
                const statusSuffix = dev.connected ? ' (LIVE CONNECTED)' : ' (NO SIGNAL)';
                optsHtml += `<option value="${dev.id}" ${isSelected ? 'selected' : ''}>${badge}${dev.label}${statusSuffix}</option>`;
            });
            optsHtml += '</optgroup>';

            // Group 2: WiFi & Network IP cameras
            if (wifiCamerasList && wifiCamerasList.length > 0) {
                optsHtml += '<optgroup label="WiFi / Network IP Cameras">';
                wifiCamerasList.forEach(w => {
                    const isSelected = (assignedSource === w.url || assignedSource === w.id);
                    optsHtml += `<option value="${w.url}" ${isSelected ? 'selected' : ''}>📡 ${w.name} (${w.url})</option>`;
                });
                optsHtml += '</optgroup>';
            }

            // Group 3: Add Camera Option
            optsHtml += '<optgroup label="Add New Camera">';
            optsHtml += '<option value="__add_wifi__">➕ Add WiFi / IP Camera (RTSP/HTTP)...</option>';
            optsHtml += '</optgroup>';

            selectEl.innerHTML = optsHtml;
            if (assignedSource) {
                selectEl.value = assignedSource;
            }

            // Update assigned text label
            updateScreenAssignedBadge(cam, assignedSource);
        });

        // Set initial UI state
        if (isCameraPowerOn) {
            setLiveUIState(Boolean(data.is_live));
        }

    } catch (e) {
        console.error('Failed to load available cameras:', e);
    }
}

let cameraPollInterval = null;
function startCameraAutoDiscovery() {
    if (cameraPollInterval) clearInterval(cameraPollInterval);
    cameraPollInterval = setInterval(() => {
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (activeTag !== 'select') {
            loadAvailableCameras(false);
        }
    }, 7000);
}

function updateScreenAssignedBadge(cam, source) {
    const badgeEl = document.getElementById('assignedLabel' + capitalize(cam));
    if (!badgeEl) return;

    if (!source || source.startsWith('video') || source.endsWith('.mp4')) {
        badgeEl.textContent = `Assigned: Demo File (${cam}.mp4)`;
    } else if (source.startsWith('http://') || source.startsWith('https://') || source.startsWith('rtsp://')) {
        const w = wifiCamerasList.find(c => c.url === source);
        badgeEl.textContent = `Assigned: 📡 ${w ? w.name : 'WiFi Camera'}`;
    } else {
        const dev = availableDevicesList.find(d => d.id === String(source));
        const devName = dev ? dev.label : `Camera ${source}`;
        badgeEl.textContent = `Assigned: ${devName}`;
    }
}

async function assignScreenCamera(screen, source) {
    if (source === '__add_wifi__') {
        openWifiModal(screen);
        const currentSrc = (currentCameraMapping[screen] && currentCameraMapping[screen].source !== undefined)
            ? String(currentCameraMapping[screen].source)
            : '';
        const sel = document.getElementById('selectSrc' + capitalize(screen));
        if (sel) sel.value = currentSrc;
        return;
    }
    try {
        const res = await fetch('/api/cameras/assign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ screen, source })
        });
        const data = await res.json();
        if (data.success) {
            currentCameraMapping[screen] = { source, connected: true };
            updateScreenAssignedBadge(screen, source);
            // Reload stream immediately on that screen box if cameras are powered on
            if (isCameraPowerOn) {
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

function openWifiModal(forScreen = null) {
    pendingWifiScreenTarget = forScreen;
    const m = document.getElementById('wifiModal');
    if (m) m.style.display = 'flex';
}

function closeWifiModal() {
    const m = document.getElementById('wifiModal');
    if (m) m.style.display = 'none';
    pendingWifiScreenTarget = null;
}

async function submitAddWifiCam() {
    const nameInput = document.getElementById('wifiCamName');
    const urlInput = document.getElementById('wifiCamUrl');
    const name = nameInput ? nameInput.value.trim() : '';
    const url = urlInput ? urlInput.value.trim() : '';
    if (!url) {
        alert('Please enter a camera stream URL (e.g. http://192.168.1.50:8080/video or rtsp://...)');
        return;
    }

    try {
        const res = await fetch('/api/cameras/wifi/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name || 'WiFi Camera', url })
        });
        const data = await res.json();
        if (data.success) {
            closeWifiModal();
            if (nameInput) nameInput.value = '';
            if (urlInput) urlInput.value = '';
            await loadAvailableCameras(true);
            if (pendingWifiScreenTarget) {
                await assignScreenCamera(pendingWifiScreenTarget, url);
            }
        } else {
            alert('Failed to add WiFi camera: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        console.error('Error adding wifi camera:', e);
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
            const statusTxt = document.getElementById('statusText' + capitalize(cam));
            if (statusTxt) {
                statusTxt.textContent = 'ONLINE';
                statusTxt.style.color = 'var(--success)';
            }
            const dot = document.getElementById('dot' + capitalize(cam));
            if (dot) dot.classList.add('active');
        });
    } else {
        if (powerInd) powerInd.classList.remove('active');
        if (powerText) powerText.textContent = 'CAMERAS OFF (STANDBY)';
        CAMERAS.forEach(cam => {
            const statusTxt = document.getElementById('statusText' + capitalize(cam));
            if (statusTxt) {
                statusTxt.textContent = 'OFFLINE';
                statusTxt.style.color = 'var(--text-muted)';
            }
            const dot = document.getElementById('dot' + capitalize(cam));
            if (dot) dot.classList.remove('active');
        });
    }
}

// ─── Camera Power Toggle (Single Toggle for All 4 Cameras) ──

async function toggleCameraPower() {
    isCameraPowerOn = !isCameraPowerOn;
    const btn = document.getElementById('btnCameraToggle');
    const icon = document.getElementById('cameraToggleIcon');
    const text = document.getElementById('cameraToggleText');
    const grid = document.getElementById('cameraGrid');

    if (isCameraPowerOn) {
        // TURN CAMERAS ON
        if (btn) {
            btn.className = 'btn-camera-toggle power-on';
        }
        if (icon) icon.className = 'fas fa-video';
        if (text) text.textContent = 'CAMERAS: ON (ALL 4 ACTIVE)';
        if (grid) grid.classList.remove('cameras-off');

        try {
            await fetch('/api/capture/start', { method: 'POST' });
        } catch (e) {
            console.error('Failed to start camera streams:', e);
        }

        // Restore image feeds with fresh timestamp
        const now = Date.now();
        CAMERAS.forEach(cam => {
            const img = document.getElementById('feed' + capitalize(cam));
            if (img) {
                img.style.display = 'block';
                img.src = `/api/capture/stream/${cam}?t=${now}`;
            }
            const statusTxt = document.getElementById('statusText' + capitalize(cam));
            if (statusTxt) {
                statusTxt.textContent = 'ONLINE';
                statusTxt.style.color = 'var(--success)';
            }
            const dot = document.getElementById('dot' + capitalize(cam));
            if (dot) dot.classList.add('active');
        });

        setLiveUIState(true);
        startStatusPolling();
    } else {
        // TURN CAMERAS OFF (Make all 4 visual screens pitch black)
        if (btn) {
            btn.className = 'btn-camera-toggle power-off';
        }
        if (icon) icon.className = 'fas fa-video-slash';
        if (text) text.textContent = 'CAMERAS: OFF (CLICK TO TURN ON)';
        if (grid) grid.classList.add('cameras-off');

        try {
            await fetch('/api/capture/stop', { method: 'POST' });
        } catch (e) {
            console.error('Failed to stop camera streams:', e);
        }

        // Make all 4 screens pure black
        CAMERAS.forEach(cam => {
            const img = document.getElementById('feed' + capitalize(cam));
            if (img) {
                img.src = '';
                img.style.display = 'none';
            }
            const statusTxt = document.getElementById('statusText' + capitalize(cam));
            if (statusTxt) {
                statusTxt.textContent = 'POWERED OFF';
                statusTxt.style.color = 'var(--text-muted)';
            }
            const dot = document.getElementById('dot' + capitalize(cam));
            if (dot) dot.classList.remove('active');
            const conf = document.getElementById('conf' + capitalize(cam));
            if (conf) conf.textContent = '—';
            const motion = document.getElementById('motion' + capitalize(cam));
            if (motion) motion.textContent = 'OFF';
        });

        setLiveUIState(false);
        updateStateIndicator('OFFLINE', 'CAMERAS POWERED OFF (SCREENS BLACK)');
    }
}

// ─── AutoCapture Toggle ────────────────────────────────────

async function toggleAutoCapture() {
    isAutoCaptureActive = !isAutoCaptureActive;
    const btn = document.getElementById('btnAutoCaptureToggle');
    const icon = document.getElementById('autoCaptureIcon');
    const text = document.getElementById('autoCaptureText');

    if (isAutoCaptureActive) {
        if (btn) btn.className = 'btn-autocapture-toggle active';
        if (icon) icon.className = 'fas fa-bolt';
        if (text) text.textContent = 'AUTOCAPTURE: ACTIVE';
    } else {
        if (btn) btn.className = 'btn-autocapture-toggle disabled-mode';
        if (icon) icon.className = 'fas fa-pause-circle';
        if (text) text.textContent = 'AUTOCAPTURE: PAUSED';
    }

    try {
        await fetch('/api/capture/set-autocapture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: isAutoCaptureActive })
        });
    } catch (e) {
        console.error('Failed to set autocapture:', e);
    }
}

// ─── Manual Trigger / Test Validity & Capture Now ──────────

async function triggerValidityCapture() {
    if (!isCameraPowerOn) {
        alert('Cameras are powered OFF. Click the CAMERAS toggle button to turn them on first.');
        return;
    }

    const btn = document.getElementById('btnTriggerCapture');
    const origHtml = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> CAPTURING...';
    }

    try {
        const res = await fetch('/api/capture/trigger', { method: 'POST' });
        const data = await res.json();

        if (data.success && data.pending) {
            if (typeof showConfirmModal === 'function') {
                showConfirmModal(data.pending);
            }
        } else {
            alert('Capture trigger failed: ' + (data.error || 'No vehicle images available.'));
        }
    } catch (e) {
        console.error('Trigger capture error:', e);
        alert('Failed to trigger capture');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHtml;
        }
    }
}

// ─── Capture Delay Setting ─────────────────────────────────

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

// ─── Screen View Switching (All 4 Screens / Front / Back / Left / Right) ─

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
        // Show all 4 visual boxes in grid
        if (grid) grid.classList.remove('single-view');
        for (const panel of Object.values(panels)) {
            if (panel) {
                panel.style.display = 'block';
                panel.classList.remove('hidden-panel');
                panel.classList.remove('single-focused');
            }
        }
    } else {
        // Show only the selected visual box; make the other 3 visual boxes completely invisible
        if (grid) grid.classList.add('single-view');
        for (const [key, panel] of Object.entries(panels)) {
            if (panel) {
                if (key === cam) {
                    panel.style.display = 'block';
                    panel.classList.remove('hidden-panel');
                    panel.classList.add('single-focused');
                } else {
                    panel.style.display = 'none';
                    panel.classList.add('hidden-panel');
                    panel.classList.remove('single-focused');
                }
            }
        }
    }
}

// ─── Status Polling & Validity State Updates ───────────────

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
    if (!isCameraPowerOn) return;

    try {
        const res = await fetch('/api/capture/status');
        const data = await res.json();

        // Update state indicator
        const cs = data.capture_state || {};
        updateStateIndicator(cs.state, cs.message);

        // Sync AutoCapture setting from backend if returned
        if (data.auto_capture !== undefined) {
            isAutoCaptureActive = Boolean(data.auto_capture);
            const btn = document.getElementById('btnAutoCaptureToggle');
            const icon = document.getElementById('autoCaptureIcon');
            const text = document.getElementById('autoCaptureText');
            if (btn && icon && text) {
                if (isAutoCaptureActive) {
                    btn.className = 'btn-autocapture-toggle active';
                    icon.className = 'fas fa-bolt';
                    text.textContent = 'AUTOCAPTURE: ACTIVE';
                } else {
                    btn.className = 'btn-autocapture-toggle disabled-mode';
                    icon.className = 'fas fa-pause-circle';
                    text.textContent = 'AUTOCAPTURE: PAUSED';
                }
            }
        }

        // Update camera detections and confidence badges
        const conds = data.conditions || {};
        for (const cam of CAMERAS) {
            const c = conds[cam] || {};

            // Detection dot & status
            const dot = document.getElementById('dot' + capitalize(cam));
            const statusText = document.getElementById('statusText' + capitalize(cam));
            if (dot && statusText) {
                if (c.detected) {
                    dot.classList.add('active');
                    statusText.textContent = 'TRACKING';
                    statusText.style.color = 'var(--success)';
                } else {
                    dot.classList.remove('active');
                    statusText.textContent = 'SEARCHING';
                    statusText.style.color = 'var(--text-muted)';
                }
            }

            // Confidence
            const conf = document.getElementById('conf' + capitalize(cam));
            if (conf) {
                conf.textContent = c.detected ? `${(c.confidence * 100).toFixed(1)}%` : '—';
            }

            // Motion
            const motion = document.getElementById('motion' + capitalize(cam));
            if (motion) {
                motion.textContent = c.motion_state || '—';
            }

            // Mini status dot on the Screen View selector pills
            const barDot = document.getElementById('camInd' + capitalize(cam));
            if (barDot) {
                if (c.detected) barDot.classList.add('active');
                else barDot.classList.remove('active');
            }
        }

        // Pending capture awaiting user confirmation & database push
        if (data.pending_capture && typeof showConfirmModal === 'function') {
            showConfirmModal(data.pending_capture);
        }

        // Capture success (after user confirmed and committed to database)
        if (data.captured && data.capture_result) {
            showCaptureSuccess(data.capture_result, data.plate_text);
        }

    } catch (e) {
        // Silent — retry next cycle
    }
}

function updateStateIndicator(state, message) {
    const dot = document.getElementById('stateDot');
    const text = document.getElementById('stateText');
    if (text) text.textContent = message || state;
    if (!dot) return;

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
        case 'AWAITING_CONFIRMATION':
        case 'CONFIRM_PENDING':
            dot.classList.add('capturing'); break;
        case 'CAPTURED':
        case 'SAVED':
            dot.classList.add('saved'); break;
        case 'ERROR':
            dot.classList.add('error'); break;
    }
}

function showCaptureSuccess(result, plateText) {
    const el = document.getElementById('captureSuccess');
    if (!el) return;
    el.style.display = 'block';

    const plate = plateText || result.vehicle_number || '';
    const isUnknown = (!plate || plate === 'UNKNOWN');
    const plateDisplay = isUnknown ? 'REGISTRATION NOT CONFIRMED' : plate;
    const plateVal = isUnknown ? '' : plate;

    const plateEl = document.getElementById('successPlate');
    if (plateEl) {
        plateEl.textContent = plateDisplay;
        if (isUnknown) {
            plateEl.style.color = 'var(--warning)';
            plateEl.style.fontSize = '20px';
        } else {
            plateEl.style.color = 'var(--text-primary)';
            plateEl.style.fontSize = '28px';
        }
    }

    const metaEl = document.getElementById('successMeta');
    if (metaEl) {
        metaEl.textContent = `${result.capture_date} — ${result.capture_time_12h}`;
    }

    const inputEl = document.getElementById('plateInput');
    if (inputEl) {
        inputEl.value = plateVal;
        if (isUnknown) {
            inputEl.placeholder = 'ENTER REGISTRATION (e.g. KL 01 AL 0009)';
        }
    }

    // Preview images
    const preview = document.getElementById('capturePreview');
    if (preview) {
        preview.innerHTML = CAMERAS.map(cam =>
            `<img src="/api/vehicle/thumbnail/${result.session_id}/${cam}"
                  onerror="this.style.background='var(--bg-surface-3)'" alt="${cam}">`
        ).join('');
    }

    const sessionLink = document.getElementById('linkViewSession');
    if (sessionLink) {
        sessionLink.href = `/vehicle/${result.session_id}`;
    }
}

async function savePlate() {
    const input = document.getElementById('plateInput');
    const plate = input ? input.value.trim() : '';
    if (!plate) return;

    try {
        const res = await fetch('/api/capture/set-plate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plate })
        });
        const data = await res.json();
        if (data.success) {
            const plateEl = document.getElementById('successPlate');
            if (plateEl) plateEl.textContent = data.plate;
        }
    } catch (e) {
        console.error('Set plate error:', e);
    }
}

function capitalize(s) {
    if (!s) return '';
    return s.charAt(0).toUpperCase() + s.slice(1);
}

// ─── Initializer on Page Load ──────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Set default view to 'all' (all 4 screens visible in 2x2 grid)
    selectCameraView('all');

    // 2. Discover available cameras and assign screen pickers
    await loadAvailableCameras(false);
    startCameraAutoDiscovery();

    // 3. Load saved capture delay
    try {
        const res = await fetch('/api/settings');
        const cfg = await res.json();
        if (cfg.capture_delay_seconds !== undefined) {
            const sel = document.getElementById('selectCaptureDelay');
            if (sel) sel.value = parseFloat(cfg.capture_delay_seconds).toFixed(1);
        }
    } catch (e) {}

    // 4. Ensure cameras are live and start status polling
    try {
        const sRes = await fetch('/api/capture/status');
        const sData = await sRes.json();
        if (sData.is_running) {
            setLiveUIState(true);
            startStatusPolling();
        } else {
            // Auto-start live view
            await fetch('/api/capture/start', { method: 'POST' });
            setLiveUIState(true);
            startStatusPolling();
        }
    } catch (e) {
        try {
            await fetch('/api/capture/start', { method: 'POST' });
            setLiveUIState(true);
            startStatusPolling();
        } catch (err) {}
    }
});
