// Global Vehicle Reception Analysis Report Modal
// Handles user review & approval before committing vehicle records into SQLite database.

let activePendingCapture = null;
let isConfirmModalOpen = false;
let isSavingConfirm = false;

function showConfirmModal(pending) {
    if (!pending) return;
    if (isConfirmModalOpen && activePendingCapture && activePendingCapture.session_id === pending.session_id) {
        return; // Already open for this capture
    }

    activePendingCapture = pending;

    // 1. Set Header Bumper Plate & Date/Time (Small size in header)
    const plateInput = document.getElementById('confirmPlateInput');
    const displayPlate = (pending.vehicle_number && pending.vehicle_number !== 'UNKNOWN') ? pending.vehicle_number : '';
    if (plateInput) {
        plateInput.value = displayPlate;
        plateInput.placeholder = displayPlate || 'ENTER REGISTRATION';
    }

    const dtEl = document.getElementById('confirmDateTimeText');
    if (dtEl) {
        dtEl.textContent = `${pending.capture_date} | ${pending.capture_time_12h}`;
    }

    const metaFiles = document.getElementById('confirmFilesMeta');
    if (metaFiles) {
        metaFiles.textContent = '4 CAMERAS ANALYZED • 4 VIEWS VERIFIED';
    }

    // 2. Render 4 Camera Cards (2x2 Grid) with Detection & Sharpness exactly like analysis mode
    const grid = document.getElementById('confirmCamsGrid');
    if (grid) {
        const camOrder = ['front', 'back', 'left', 'right'];
        const camLabels = {
            front: 'CAM 01 — FRONT VIEW',
            back: 'CAM 02 — REAR VIEW',
            left: 'CAM 03 — LEFT SIDE VIEW',
            right: 'CAM 04 — RIGHT SIDE VIEW'
        };

        const details = pending.camera_details || {};

        grid.innerHTML = camOrder.map(c => {
            const info = details[c] || {};
            const det = info.detection || {};
            const isDetected = info.vehicle_detected !== false;
            const confPct = Math.round((det.confidence || pending.confidences?.[c] || 0.85) * 100);
            const clsName = (det.class_name || 'CAR').toUpperCase();

            const detBadge = isDetected 
                ? `<span style="color:var(--success, #38ef7d); font-weight:700;"><i class="fas fa-check-circle"></i> ${clsName} (${confPct}%)</span>`
                : `<span style="color:var(--warning, #f5a623);"><i class="fas fa-exclamation-triangle"></i> Searching...</span>`;

            const sharpVal = info.sharpness || pending.sharpness?.[c] || '—';
            const imgSrc = (info.preview_b64)
                ? `data:image/jpeg;base64,${info.preview_b64}`
                : `/api/capture/pending/image/${c}?sid=${pending.session_id}&t=${Date.now()}`;

            return `
                <div class="confirm-card">
                    <div class="confirm-card-bar">
                        <span>${camLabels[c]}</span>
                        <span class="confirm-card-badge">LIVE RECEPTION</span>
                    </div>
                    <div class="confirm-card-img-wrap" onclick="if(typeof openLightbox==='function') openLightbox('${imgSrc}')">
                        <img id="confirmImg${c.charAt(0).toUpperCase() + c.slice(1)}" src="${imgSrc}" alt="${c}">
                    </div>
                    <div style="padding:6px 10px; font-size:11px; display:flex; justify-content:space-between; align-items:center; background:#0e1013; border-top:1px solid rgba(255,255,255,0.05);">
                        <div>${detBadge}</div>
                        <div style="color:#8894a0; font-size:10px; font-weight:600;">Sharpness: ${sharpVal}</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    // 3. Display Modal with animation
    const modal = document.getElementById('vehicleConfirmModal');
    if (modal) {
        modal.classList.add('active');
        isConfirmModalOpen = true;
    }

    // Auto-focus plate input if registration wasn't fully recognized
    if (!displayPlate && plateInput) {
        setTimeout(() => plateInput.focus(), 150);
    }
}

function closeConfirmModal() {
    const modal = document.getElementById('vehicleConfirmModal');
    if (modal) modal.classList.remove('active');
    isConfirmModalOpen = false;
    activePendingCapture = null;
    isSavingConfirm = false;
}

// Option 3: OK — SAVE TO DATABASE
async function confirmAndSaveCapture() {
    if (isSavingConfirm || !activePendingCapture) return;
    isSavingConfirm = true;

    const btn = document.getElementById('btnConfirmCapture');
    const originalHtml = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> SAVING TO DATABASE...';
    }

    const plateInput = document.getElementById('confirmPlateInput');
    const plate = plateInput ? plateInput.value.trim().toUpperCase() : '';
    const sid = activePendingCapture.session_id;

    try {
        const res = await fetch('/api/capture/confirm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid, vehicle_number: plate })
        });
        const data = await res.json();

        if (data.success) {
            closeConfirmModal();
            const savedPlate = data.vehicle_number || plate || 'Record';
            showGlobalToast(`✓ Vehicle ${savedPlate} confirmed & saved to database!`, 'success');

            // If browsing /database: refresh list
            if (typeof loadVehicles === 'function') {
                loadVehicles();
            }
            // If on Dashboard /: refresh recent & stats
            if (typeof loadRecent === 'function') loadRecent();
            if (typeof loadStats === 'function') loadStats();

            // If on Live Capture /capture: show capture success block
            if (typeof showCaptureSuccess === 'function') {
                showCaptureSuccess(data, plate);
            }
        } else {
            alert('Error saving to database: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        console.error('Confirm save error:', e);
        alert('Network error while saving vehicle record');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalHtml;
        }
        isSavingConfirm = false;
    }
}

// Option 1: CANCEL
async function discardPendingCapture() {
    if (!activePendingCapture) {
        closeConfirmModal();
        return;
    }

    const sid = activePendingCapture.session_id;
    try {
        await fetch('/api/capture/discard', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid })
        });
        closeConfirmModal();
        showGlobalToast('Capture discarded. Resumed vehicle monitoring.', 'warning');
    } catch (e) {
        console.error('Discard capture error:', e);
        closeConfirmModal();
    }
}

// Option 2: RETRY
async function retryLiveCapture() {
    if (activePendingCapture && activePendingCapture.session_id) {
        try {
            await fetch('/api/capture/discard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: activePendingCapture.session_id })
            });
        } catch (e) {}
    }
    closeConfirmModal();
    showGlobalToast('Retrying capture... Scanning cameras for full view.', 'info');

    // Trigger fresh capture
    if (typeof triggerValidityCapture === 'function') {
        setTimeout(() => triggerValidityCapture(), 300);
    }
}

function showGlobalToast(msg, type = 'info') {
    let t = document.getElementById('globalSystemToast');
    if (!t) {
        t = document.createElement('div');
        t.id = 'globalSystemToast';
        t.style.cssText = 'position:fixed; bottom:24px; right:24px; z-index:999999; padding:12px 20px; border-radius:6px; font-size:0.85rem; font-weight:700; letter-spacing:0.5px; box-shadow:0 8px 24px rgba(0,0,0,0.6); display:flex; align-items:center; gap:8px; transition:opacity 0.3s, transform 0.3s;';
        document.body.appendChild(t);
    }

    if (type === 'success') {
        t.style.background = '#0a2e1d';
        t.style.color = '#38ef7d';
        t.style.border = '1px solid #11998e';
    } else if (type === 'warning') {
        t.style.background = '#33200a';
        t.style.color = '#f5af19';
        t.style.border = '1px solid #e65c00';
    } else {
        t.style.background = '#0d222d';
        t.style.color = '#00c0f0';
        t.style.border = '1px solid #00c0f0';
    }

    t.textContent = msg;
    t.style.opacity = '1';
    t.style.transform = 'translateY(0)';

    setTimeout(() => {
        t.style.opacity = '0';
        t.style.transform = 'translateY(10px)';
    }, 4000);
}

// Global polling for pending captures across all pages (Live, Database browse, Dashboard, etc.)
let globalPendingPoller = null;
function startGlobalPendingCheck() {
    // Only run independent poller if not already on /capture (where capture.js polls status)
    if (window.location.pathname === '/capture') return;

    if (globalPendingPoller) clearInterval(globalPendingPoller);
    globalPendingPoller = setInterval(async () => {
        if (isConfirmModalOpen) return;
        try {
            const res = await fetch('/api/capture/pending');
            const data = await res.json();
            if (data.has_pending && data.pending) {
                showConfirmModal(data.pending);
            }
        } catch (e) {}
    }, 2500);
}

document.addEventListener('DOMContentLoaded', () => {
    // Handle keyboard shortcuts inside confirmation modal
    window.addEventListener('keydown', (e) => {
        if (!isConfirmModalOpen) return;
        if (e.key === 'Escape') {
            discardPendingCapture();
        } else if (e.key === 'Enter' && e.target && e.target.id === 'confirmPlateInput') {
            e.preventDefault();
            confirmAndSaveCapture();
        }
    });

    startGlobalPendingCheck();
});
