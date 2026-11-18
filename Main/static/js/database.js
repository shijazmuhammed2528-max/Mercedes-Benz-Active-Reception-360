// Vehicle Database Search JavaScript — Mercedes-Benz Active Reception

document.addEventListener('DOMContentLoaded', () => {
    // Perform initial search to show recent records
    searchDatabase();

    // Allow Enter key in search input
    const searchInput = document.getElementById('searchNumber');
    if (searchInput) {
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                searchDatabase();
            }
        });
    }
});

async function searchDatabase() {
    const vehicleNumber = document.getElementById('searchNumber').value.trim();
    const date = document.getElementById('searchDate').value.trim();
    const timeFrom = document.getElementById('searchTimeFrom').value.trim();
    const timeTo = document.getElementById('searchTimeTo').value.trim();

    const payload = {
        vehicle_number: vehicleNumber,
        date: date,
        time_from: timeFrom,
        time_to: timeTo
    };

    const countEl = document.getElementById('resultCount');
    const tableEl = document.getElementById('resultsTable');
    const bodyEl = document.getElementById('resultsBody');
    const emptyEl = document.getElementById('emptyState');

    countEl.textContent = 'Searching...';

    try {
        const response = await fetch('/api/database/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const results = await response.json();

        if (!results || results.length === 0) {
            tableEl.style.display = 'none';
            emptyEl.style.display = 'flex';
            countEl.textContent = '0 records found';
            bodyEl.innerHTML = '';
            return;
        }

        countEl.textContent = `${results.length} record${results.length === 1 ? '' : 's'} found`;
        emptyEl.style.display = 'none';
        tableEl.style.display = 'table';

        bodyEl.innerHTML = results.map(row => {
            const sid = escapeHtml(row.session_id);
            const vnum = escapeHtml(row.vehicle_number || 'UNKNOWN');
            const cdate = escapeHtml(row.capture_date || '—');
            const ctime = escapeHtml(row.capture_time_12h || '—');
            const status = (row.capture_status || 'success').toLowerCase();
            const badgeClass = status === 'success' ? 'badge-success' : 'badge-warning';

            return `
                <tr>
                    <td>
                        <img class="thumbnail" 
                             src="/api/vehicle/thumbnail/${sid}/front" 
                             alt="${vnum}"
                             onerror="this.src='/static/images/placeholder.svg'; this.onerror=null;" />
                    </td>
                    <td>
                        <strong style="color:var(--text-primary);letter-spacing:0.5px;">${vnum}</strong>
                    </td>
                    <td>${cdate}</td>
                    <td>${ctime}</td>
                    <td>
                        <span class="badge ${badgeClass}">${status.toUpperCase()}</span>
                    </td>
                    <td>
                        <a href="/vehicle/${sid}" class="btn btn-sm btn-primary">
                            <i class="fas fa-eye"></i> VIEW DETAILS
                        </a>
                    </td>
                </tr>
            `;
        }).join('');

    } catch (err) {
        console.error('Search error:', err);
        countEl.textContent = 'Error loading results';
        tableEl.style.display = 'none';
        emptyEl.style.display = 'flex';
        emptyEl.innerHTML = `
            <i class="fas fa-exclamation-triangle" style="color:var(--error);"></i>
            <h3>SEARCH FAILED</h3>
            <p>Could not load vehicle records from the database. Please try again.</p>
        `;
    }
}

function clearSearch() {
    document.getElementById('searchNumber').value = '';
    document.getElementById('searchDate').value = '';
    document.getElementById('searchTimeFrom').value = '';
    document.getElementById('searchTimeTo').value = '';
    searchDatabase();
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
