// Global Vehicle Reception Confirmation Modal
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

    // 2. Set 4 Vehicle Camera Images
    const cams = ['front', 'back', 'left', 'right'];
    cams.forEach(cam => {
        const cap = cam.charAt(0).toUpperCase() + cam.slice(1);
        const img = document.getElementById('confirmImg' + cap);
        const badge = document.getElementById('confirmBadge' + cap);

        if (img) {
            const url = (pending.images && pending.images[cam])
                ? pending.images[cam]
                : `/api/capture/pending/image/${cam}?sid=${pending.session_id}&t=${Date.now()}`;
            img.src = url;
        }

        if (badge && pending.confidences && pending.confidences[cam] !== undefined) {
            const pct = Math.round((parseFloat(pending.confidences[cam]) || 0) * 100);
            badge.textContent = pct > 0 ? `${pct}% CONFIDENCE` : 'VERIFIED';
        }
    });

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
            showGlobalToast(`✓ Vehicle ${data.vehicle_number || plate || 'Record'} confirmed & saved to database!`, 'success');

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
