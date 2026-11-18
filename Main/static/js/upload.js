// Vehicle Multi-Media Upload & Analysis JavaScript — Mercedes-Benz Active Reception

let selectedFilesList = []; // Array of { file: File, cameraView: string }
let currentUploadedSessionId = null;

const CAM_VIEWS = [
    { id: 'front', label: 'CAM 01 — FRONT VIEW' },
    { id: 'back', label: 'CAM 02 — REAR VIEW' },
    { id: 'left', label: 'CAM 03 — LEFT SIDE VIEW' },
    { id: 'right', label: 'CAM 04 — RIGHT SIDE VIEW' }
];

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = 'var(--primary, #00c0f0)';
            dropZone.style.background = 'rgba(0, 192, 240, 0.05)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = 'var(--border-light)';
            dropZone.style.background = 'rgba(255, 255, 255, 0.02)';
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleIncomingFiles(files);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleIncomingFiles(fileInput.files);
        }
    });

    updateViewCoverageTracker();
});

function handleIncomingFiles(fileList) {
    const newFiles = Array.from(fileList);
    const validMedia = newFiles.filter(f => f.type.startsWith('image/') || f.type.startsWith('video/') || isMediaExtension(f.name));

    if (validMedia.length === 0) {
        alert('Please select valid vehicle photo or video files (JPG, PNG, WEBP, MP4, MOV, AVI).');
        return;
    }

    // Combine with existing up to maximum 4
    for (const f of validMedia) {
        if (selectedFilesList.length >= 4) {
            alert('Maximum 4 media files allowed for 360° reception (Front, Rear, Left, Right).');
            break;
        }
        // Avoid adding duplicate filename
        if (!selectedFilesList.some(item => item.file.name === f.name && item.file.size === f.size)) {
            const defaultView = guessViewFromFilename(f.name, selectedFilesList.map(x => x.cameraView));
            selectedFilesList.push({
                file: f,
                cameraView: defaultView
            });
        }
    }

    renderSelectedFiles();
}

function guessViewFromFilename(filename, alreadyUsed) {
    const fn = filename.toLowerCase();
    const views = ['front', 'back', 'left', 'right'];

    if ((fn.includes('front')) && !alreadyUsed.includes('front')) return 'front';
    if ((fn.includes('back') || fn.includes('rear')) && !alreadyUsed.includes('back')) return 'back';
    if ((fn.includes('left')) && !alreadyUsed.includes('left')) return 'left';
    if ((fn.includes('right')) && !alreadyUsed.includes('right')) return 'right';

    // Fallback to first available unused view
    for (const v of views) {
        if (!alreadyUsed.includes(v)) return v;
    }
    return 'front';
}

function isMediaExtension(name) {
    const lower = name.toLowerCase();
    return ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.jpg', '.jpeg', '.png', '.webp'].some(ext => lower.endsWith(ext));
}

function updateViewCoverageTracker() {
    const views = ['front', 'back', 'left', 'right'];

    views.forEach(v => {
        const slot = document.getElementById('slot' + capitalize(v));
        const text = document.getElementById('slotText' + capitalize(v));
        if (!slot || !text) return;

        const match = selectedFilesList.find(item => item.cameraView === v);
        if (match) {
            slot.style.background = 'rgba(0, 200, 83, 0.08)';
            slot.style.borderColor = 'var(--success)';
            text.innerHTML = `<span style="color:var(--success); font-weight:700;"><i class="fas fa-check-circle"></i> Ready</span>`;
        } else {
            slot.style.background = '#161616';
            slot.style.borderColor = '#333';
            text.innerHTML = `<span style="color:var(--warning);"><i class="fas fa-exclamation-circle"></i> Missing</span>`;
        }
    });
}

function renderSelectedFiles() {
    const section = document.getElementById('selectedFilesSection');
    const container = document.getElementById('selectedFilesContainer');
    const countBadge = document.getElementById('fileCountBadge');
    const btnAnalyze = document.getElementById('btnAnalyze');

    countBadge.textContent = selectedFilesList.length;

    updateViewCoverageTracker();

    if (selectedFilesList.length === 0) {
        section.style.display = 'none';
        btnAnalyze.disabled = true;
        container.innerHTML = '';
        return;
    }

    section.style.display = 'block';
    btnAnalyze.disabled = false;

    container.innerHTML = selectedFilesList.map((item, index) => {
        const isVid = item.file.type.startsWith('video/') || ['.mp4', '.mov', '.avi', '.mkv'].some(ext => item.file.name.toLowerCase().endsWith(ext));
        const iconClass = isVid ? 'fa-video' : 'fa-image';
        const tagText = isVid ? 'VIDEO' : 'PHOTO';
        const tagBg = isVid ? 'rgba(77, 166, 255, 0.15)' : 'rgba(0, 200, 83, 0.15)';
        const tagColor = isVid ? '#4da6ff' : '#00c853';

        const optionsHtml = CAM_VIEWS.map(v => 
            `<option value="${v.id}" ${item.cameraView === v.id ? 'selected' : ''}>${v.label}</option>`
        ).join('');

        return `
            <div style="background:#181818; border:1px solid #2a2a2a; border-radius:6px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; gap:12px;">
                <div style="display:flex; align-items:center; gap:10px; flex:1; min-width:0;">
                    <i class="fas ${iconClass}" style="color:${tagColor}; font-size:16px;"></i>
                    <div style="min-width:0; flex:1;">
                        <div style="font-weight:600; font-size:12px; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${item.file.name}">
                            ${item.file.name}
                        </div>
                        <div style="font-size:11px; color:#888; display:flex; gap:8px; margin-top:2px;">
                            <span>${formatBytes(item.file.size)}</span>
                            <span style="background:${tagBg}; color:${tagColor}; padding:1px 6px; border-radius:2px; font-weight:700; font-size:9px;">${tagText}</span>
                        </div>
                    </div>
                </div>

                <div style="display:flex; align-items:center; gap:8px;">
                    <select onchange="changeItemView(${index}, this.value)" style="background:#111; color:#fff; border:1px solid #444; padding:5px 8px; border-radius:4px; font-size:11px; font-weight:600; cursor:pointer;">
                        ${optionsHtml}
                    </select>
                    <button type="button" class="btn btn-sm" onclick="removeFileItem(${index})" style="padding:4px 8px; color:#ff5555;" title="Remove this file">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function changeItemView(index, newView) {
    if (selectedFilesList[index]) {
        selectedFilesList[index].cameraView = newView;
        updateViewCoverageTracker();
    }
}

function removeFileItem(index) {
    selectedFilesList.splice(index, 1);
    renderSelectedFiles();
}

function clearAllFiles() {
    selectedFilesList = [];
    document.getElementById('fileInput').value = '';
    renderSelectedFiles();
    document.getElementById('resultsCard').style.display = 'none';
}

function checkViewConflicts() {
    const usedViews = selectedFilesList.map(x => x.cameraView);
    const duplicates = usedViews.filter((item, index) => usedViews.indexOf(item) !== index);
    return duplicates;
}

function getMissingViews() {
    const usedViews = selectedFilesList.map(x => x.cameraView);
    return ['front', 'back', 'left', 'right'].filter(v => !usedViews.includes(v));
}

function showIncompleteModal(missing) {
    const modal = document.getElementById('incompleteModal');
    const countEl = document.getElementById('modalSelectedCount');
    const listEl = document.getElementById('modalMissingList');

    countEl.textContent = `${selectedFilesList.length}`;

    const missingMap = {
        front: 'CAM 01 — FRONT VIEW',
        back: 'CAM 02 — REAR VIEW',
        left: 'CAM 03 — LEFT SIDE VIEW',
        right: 'CAM 04 — RIGHT SIDE VIEW'
    };

    listEl.innerHTML = missing.map(m =>
        `<span style="background:rgba(245,166,35,0.15); color:var(--warning); border:1px solid rgba(245,166,35,0.4); padding:4px 10px; border-radius:4px; font-weight:700; font-size:11px;">
            <i class="fas fa-camera"></i> ${missingMap[m]}
        </span>`
    ).join('');

    modal.style.display = 'flex';
}

function addMissingFilesFromModal() {
    document.getElementById('incompleteModal').style.display = 'none';
    document.getElementById('fileInput').click();
}

function proceedAnywayFromModal() {
    document.getElementById('incompleteModal').style.display = 'none';
    executeUploadAndAnalyze();
}

async function uploadAndAnalyze() {
    if (selectedFilesList.length === 0) return;

    // Check 1: View conflict (duplicate view assignment)
    const duplicates = checkViewConflicts();
    if (duplicates.length > 0) {
        const dupNames = duplicates.map(d => d.toUpperCase()).join(', ');
        alert(`⚠️ Camera View Assignment Conflict!\n\nYou have multiple files assigned to the same camera view (${dupNames}).\n\nPlease select unique camera angles (Front, Rear, Left, Right) using the dropdown on each file.`);
        return;
    }

    // Check 2: Fewer than 4 views selected -> ASK USER FIRST
    const missing = getMissingViews();
    if (missing.length > 0) {
        showIncompleteModal(missing);
        return;
    }

    executeUploadAndAnalyze();
}

async function executeUploadAndAnalyze() {
    const btn = document.getElementById('btnAnalyze');
    const resultsCard = document.getElementById('resultsCard');
    const saveToDb = document.getElementById('checkSaveDb').checked;
    const plateOverride = document.getElementById('plateOverride').value.trim();

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> EXTRACTING FRAMES &amp; RUNNING YOLO/OCR...';

    const formData = new FormData();
    selectedFilesList.forEach((item) => {
        formData.append(item.cameraView, item.file);
    });

    formData.append('save_to_database', saveToDb ? 'true' : 'false');
    if (plateOverride) {
        formData.append('vehicle_number', plateOverride);
    }

    try {
        const response = await fetch('/api/vehicle/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-microchip"></i> RE-ANALYZE ALL VIEWS';

        if (!data.success) {
            alert('Analysis error: ' + (data.error || 'Failed to process vehicle media'));
            return;
        }

        currentUploadedSessionId = data.session_id;

        // Render Results
        resultsCard.style.display = 'block';

        // Plate Banner & Input
        const plate = data.plate_text || 'UNKNOWN';
        const plateEl = document.getElementById('resPlate');
        plateEl.textContent = plate;
        plateEl.style.color = (plate === 'UNKNOWN') ? 'var(--warning)' : 'var(--success)';

        const plateInput = document.getElementById('resPlateInput');
        if (plateInput) plateInput.value = (plate === 'UNKNOWN') ? '' : plate;

        document.getElementById('resMetaRight').innerHTML = `
            <div><strong>Total Files:</strong> ${data.total_files}</div>
            <div><strong>Status:</strong> ${data.saved ? '<span style="color:var(--success)">Stored in Data_base</span>' : 'Analysis Only'}</div>
        `;

        // Render 4-Camera Grid (2x2)
        const grid = document.getElementById('fourCamResultsGrid');
        const cams = data.cameras || {};
        const camOrder = ['front', 'back', 'left', 'right'];
        const camLabels = {
            front: 'CAM 01 — FRONT VIEW',
            back: 'CAM 02 — REAR VIEW',
            left: 'CAM 03 — LEFT SIDE VIEW',
            right: 'CAM 04 — RIGHT SIDE VIEW'
        };

        grid.innerHTML = camOrder.map(c => {
            const info = cams[c] || {};
            const det = info.detection || {};
            const isDetected = info.vehicle_detected;
            const detBadge = isDetected 
                ? `<span style="color:var(--success); font-weight:700;"><i class="fas fa-check-circle"></i> ${(det.class_name || 'CAR').toUpperCase()} (${Math.round(det.confidence * 100)}%)</span>`
                : `<span style="color:var(--warning);"><i class="fas fa-exclamation-triangle"></i> No vehicle detected</span>`;

            const formatTag = info.is_video ? `VIDEO (${info.duration_seconds}s)` : 'PHOTO';

            return `
                <div style="background:#141414; border:1px solid ${isDetected ? '#2a2a2a' : 'rgba(245,166,35,0.4)'}; border-radius:6px; overflow:hidden;">
                    <div style="padding:8px 12px; background:#1b1b1b; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #2a2a2a;">
                        <span style="font-size:11px; font-weight:700; color:#fff; letter-spacing:0.05em;">${camLabels[c]}</span>
                        <span style="font-size:9px; font-weight:700; background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:3px; color:#aaa;">${formatTag}</span>
                    </div>
                    <div style="position:relative; width:100%; height:150px; background:#000;">
                        ${info.preview_b64 ? `
                            <img src="data:image/jpeg;base64,${info.preview_b64}" style="width:100%; height:100%; object-fit:cover;" alt="${c}">
                        ` : `
                            <div style="height:100%; display:flex; align-items:center; justify-content:center; color:#555; font-size:12px;">NO SIGNAL</div>
                        `}
                    </div>
                    <div style="padding:8px 12px; font-size:11px; display:flex; justify-content:space-between; align-items:center; background:#141414;">
                        <div>${detBadge}</div>
                        <div style="color:#777; font-size:10px;">Sharpness: ${info.sharpness || '—'}</div>
                    </div>
                </div>
            `;
        }).join('');

        // Action links
        const actionsDiv = document.getElementById('savedActionBtns');
        if (data.saved && data.session_id) {
            document.getElementById('btnOpenSession').href = `/vehicle/${data.session_id}`;
            actionsDiv.style.display = 'flex';
        } else {
            actionsDiv.style.display = 'none';
        }

        // Scroll into view
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        console.error('Upload error:', err);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-microchip"></i> ANALYZE ALL VIEWS &amp; STORE VEHICLE';
        alert('Server communication error during media analysis');
    }
}

async function saveUploadPlate() {
    const input = document.getElementById('resPlateInput');
    if (!input || !currentUploadedSessionId) return;

    const newPlate = input.value.trim().toUpperCase();
    if (!newPlate) {
        alert('Please enter a valid registration number.');
        return;
    }

    try {
        const res = await fetch('/api/capture/set-plate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plate: newPlate })
        });
        const data = await res.json();
        if (data.success) {
            const plateEl = document.getElementById('resPlate');
            plateEl.textContent = data.plate;
            plateEl.style.color = 'var(--success)';
            alert(`Registration updated to: ${data.plate}`);
        } else {
            alert('Update failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        console.error('Plate update error:', e);
        alert('Failed to update plate');
    }
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
