// Dashboard JavaScript — Mercedes-Benz Active Reception

async function loadStats() {
    try {
        const res = await fetch('/api/dashboard/stats');
        const stats = await res.json();
        document.getElementById('statTotal').textContent = stats.total_sessions || 0;
        document.getElementById('statToday').textContent = stats.today_sessions || 0;
        document.getElementById('statSuccess').textContent = stats.success_sessions || 0;
        document.getElementById('statStored').textContent = stats.total_vehicles || 0;
    } catch (e) {
        console.error('Stats error:', e);
    }
}

async function loadRecent() {
    try {
        const res = await fetch('/api/dashboard/recent?limit=8');
        const sessions = await res.json();
        const container = document.getElementById('recentSessions');

        if (!sessions || sessions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-car-side"></i>
                    <p class="text-muted">No captures yet. Start a new capture session.</p>
                </div>`;
            return;
        }

        container.innerHTML = sessions.map(s => `
            <a href="/vehicle/${s.session_id}" class="session-item">
                <img class="session-thumb" src="/api/vehicle/thumbnail/${s.session_id}/front"
                     onerror="this.style.display='none'" alt="">
                <div class="session-info">
                    <div class="vehicle-num">${s.vehicle_number || 'UNKNOWN'}</div>
                    <div class="session-time">${s.capture_date} — ${s.capture_time_12h}</div>
                </div>
                <span class="badge badge-${s.capture_status === 'success' ? 'success' : 'warning'}">
                    ${(s.capture_status || 'unknown').toUpperCase()}
                </span>
            </a>
        `).join('');
    } catch (e) {
        console.error('Recent error:', e);
    }
}

async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        const videos = data.videos || {};
        for (const [name, info] of Object.entries(videos)) {
            const el = document.getElementById('status' + name.charAt(0).toUpperCase() + name.slice(1));
            if (el) {
                const dot = el.querySelector('.status-dot');
                dot.style.background = info.exists ? 'var(--success)' : 'var(--error)';
            }
        }

        const yoloDot = document.querySelector('#statusYolo .status-dot');
        yoloDot.style.background = data.yolo_ready ? 'var(--success)' : 'var(--warning)';

        const dbDot = document.querySelector('#statusDb .status-dot');
        dbDot.style.background = data.database ? 'var(--success)' : 'var(--error)';
    } catch (e) {
        console.error('Status error:', e);
    }
}

// Initial load
loadStats();
loadRecent();
loadStatus();

// Auto-refresh
setInterval(() => {
    loadStats();
    loadRecent();
}, 10000);
