// Vehicle Image Upload & Analysis JavaScript — Mercedes-Benz Active Reception

let currentSelectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    if (!dropZone || !fileInput) return;

    // Drag and drop events
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
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, WEBP).');
        return;
    }

    currentSelectedFile = file;

    // Show preview & file info
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('localPreviewImg').src = e.target.result;
        document.getElementById('selectedFileName').textContent = file.name;
        document.getElementById('selectedFileSize').textContent = formatBytes(file.size);

        document.getElementById('dropPrompt').style.display = 'none';
        document.getElementById('fileSelectedInfo').style.display = 'block';
        document.getElementById('btnAnalyze').disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearSelectedFile(e) {
    if (e) e.stopPropagation();
    currentSelectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('localPreviewImg').src = '';
    document.getElementById('dropPrompt').style.display = 'block';
    document.getElementById('fileSelectedInfo').style.display = 'none';
    document.getElementById('btnAnalyze').disabled = true;
    document.getElementById('resultsCard').style.display = 'none';
}

async function uploadAndAnalyze() {
    if (!currentSelectedFile) return;

    const btn = document.getElementById('btnAnalyze');
    const resultsCard = document.getElementById('resultsCard');
    const cameraView = document.getElementById('selectView').value;
    const saveToDb = document.getElementById('checkSaveDb').checked;
    const plateOverride = document.getElementById('plateOverride').value.trim();

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> RUNNING YOLO &amp; OCR ANALYSIS...';

    const formData = new FormData();
    formData.append('file', currentSelectedFile);
    formData.append('camera_view', cameraView);
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
        btn.innerHTML = '<i class="fas fa-microchip"></i> RE-ANALYZE &amp; STORE';

        if (!data.success) {
            alert('Analysis error: ' + (data.error || 'Failed to process vehicle image'));
            return;
        }

        // Render Results
        resultsCard.style.display = 'block';

        // Annotated Image with green YOLO bounding box
        if (data.preview_b64) {
            document.getElementById('annotatedImg').src = 'data:image/jpeg;base64,' + data.preview_b64;
        }

        // Detected Class
        const det = data.detection || {};
        const detClass = (det.class_name || 'Vehicle').toUpperCase();
        document.getElementById('resClass').textContent = data.vehicle_detected ? detClass : 'No Vehicle Detected';

        // Confidence
        const confPercent = det.confidence ? `${(det.confidence * 100).toFixed(1)}%` : '—';
        document.getElementById('resConf').textContent = confPercent;

        // Plate
        const plate = data.plate_text || 'UNKNOWN';
        const plateEl = document.getElementById('resPlate');
        plateEl.textContent = plate;
        plateEl.style.color = (plate === 'UNKNOWN') ? 'var(--warning)' : 'var(--success)';

        // Sharpness & Resolution
        document.getElementById('resSharpness').textContent = `${data.sharpness} ${data.sharpness >= 30 ? '(Sharp)' : '(Low)'}`;
        document.getElementById('resResolution').textContent = `${data.width} × ${data.height} px`;

        // DB Status & Action Links
        const dbStatusEl = document.getElementById('resDbStatus');
        const actionsDiv = document.getElementById('savedActionBtns');

        if (data.saved && data.session_id) {
            dbStatusEl.innerHTML = `<span style="color:var(--success); font-weight:600;"><i class="fas fa-check-circle"></i> Saved in Data_base</span>`;
            document.getElementById('btnOpenSession').href = `/vehicle/${data.session_id}`;
            actionsDiv.style.display = 'flex';
        } else {
            dbStatusEl.textContent = 'Analysis only (Not saved to database)';
            actionsDiv.style.display = 'none';
        }

        // Scroll to results smoothly
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        console.error('Upload error:', err);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-microchip"></i> ANALYZE &amp; STORE VEHICLE';
        alert('Server communication error during image analysis');
    }
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
