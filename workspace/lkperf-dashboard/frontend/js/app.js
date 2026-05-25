const loadingEl = document.getElementById('globalLoading');
const datePanel = document.getElementById('datePickerPanel');
const dayPanel = document.getElementById('dayPanel');
const backBtn = document.getElementById('backBtn');

let currentDate = null;
let sliderStartMin = 0;
let sliderEndMin = 1439;
let isDragging = null;
let rawOffset = 0;
let rawPageSize = 50;
let hourlyData = [];
let dropdownData = { name: [], tag: [], room: [] };
let timelineData = [];
let timelineLayout = [];
let tlScale = 1;
let currentTz = localStorage.getItem('lkperf-tz') || 'UTC';

/* ========== 时区工具 ========== */
function getTzOffsetMinutes(tz) {
    if (tz === 'UTC') return 0;
    if (tz === 'local') return -new Date().getTimezoneOffset();
    const now = new Date();
    const utcStr = now.toLocaleString('en-US', { timeZone: 'UTC' });
    const tzStr  = now.toLocaleString('en-US', { timeZone: tz });
    return Math.round((new Date(tzStr).getTime() - new Date(utcStr).getTime()) / 60000);
}

function formatLocal(us) {
    if (!us) return '-';
    const ms = us / 1000;
    if (currentTz === 'UTC') {
        const d = new Date(ms);
        const h = d.getUTCHours().toString().padStart(2,'0');
        const m = d.getUTCMinutes().toString().padStart(2,'0');
        const s = d.getUTCSeconds().toString().padStart(2,'0');
        return `${h}:${m}:${s}`;
    }
    if (currentTz === 'local') {
        return new Date(ms).toLocaleTimeString();
    }
    try {
        return new Date(ms).toLocaleTimeString('zh-CN', {
            timeZone: currentTz, hour12: false,
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    } catch (e) {
        const d = new Date(ms);
        return `${d.getUTCHours().toString().padStart(2,'0')}:${d.getUTCMinutes().toString().padStart(2,'0')}:${d.getUTCSeconds().toString().padStart(2,'0')}`;
    }
}

function onTzChange() {
    const sel = document.getElementById('tzSelect');
    if (!sel) return;
    currentTz = sel.value;
    localStorage.setItem('lkperf-tz', currentTz);
    const hint = document.getElementById('tzHint');
    if (hint) {
        if (currentTz === 'UTC') hint.textContent = '当前按 UTC 显示';
        else if (currentTz === 'local') hint.textContent = '当前按本地时间显示';
        else hint.textContent = `当前按 ${currentTz} 显示`;
    }
    loadTraces();
    loadRawSpans();
    refreshStats();
    renderTimeline();
    const treeContainer = document.getElementById('traceTreeContainer');
    if (treeContainer && treeContainer.style.display !== 'none') {
        const rootText = document.getElementById('treeRootName')?.textContent || '';
        const m = rootText.match(/\[([^\]]+)\]/);
        if (m) showTraceTree(m[1]);
    }
}

/* ========== Timeline 缩放控制 ========== */
function updateTlZoomUI() {
    const slider = document.getElementById('tlZoomSlider');
    const text = document.getElementById('tlZoomText');
    if (!slider || !text) return;
    const pct = Math.max(10, Math.min(1000, Math.round(tlScale * 100)));
    slider.value = pct;
    text.textContent = pct + '%';
}

function onTlZoomSlide(val) {
    tlScale = Math.max(0.01, val / 100);
    const text = document.getElementById('tlZoomText');
    if (text) text.textContent = Math.round(tlScale * 100) + '%';
    renderTimeline();
}

function setTlZoom(mult) {
    tlScale *= mult;
    tlScale = Math.max(0.01, tlScale);
    updateTlZoomUI();
    renderTimeline();
}

function resetTlZoom() {
    tlScale = 1;
    updateTlZoomUI();
    renderTimeline();
}

/* ========== 基础 UI ========== */
function showLoading(text) {
    document.querySelector('.loading-text').textContent = text;
    loadingEl.style.display = 'flex';
}
function hideLoading() { loadingEl.style.display = 'none'; }

function initTheme() {
    const saved = localStorage.getItem('lkperf-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    document.getElementById('themeBtn').textContent = saved === 'dark' ? '🌙' : '☀️';
    const tzSel = document.getElementById('tzSelect');
    if (tzSel) tzSel.value = currentTz;
}
function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('lkperf-theme', next);
    document.getElementById('themeBtn').textContent = next === 'dark' ? '🌙' : '☀️';
}

async function loadDates() {
    showLoading('正在扫描日志目录...');
    try {
        const res = await fetch('/api/dates');
        const dates = await res.json();
        const grid = document.getElementById('dateGrid');
        if (!dates.length) { grid.innerHTML = '<div class="loading-inline">暂无日志数据</div>'; return; }
        const maxLines = Math.max(...dates.map(d => d.total_lines), 1);
        grid.innerHTML = dates.map(d => {
            const fmt = `${d.date.slice(0,4)}-${d.date.slice(4,6)}-${d.date.slice(6,8)}`;
            const pct = (d.total_lines / maxLines) * 100;
            return `<div class="date-card" onclick="selectDate('${d.date}')">
                <div class="date-label">${fmt}</div>
                <div class="date-meta">${d.total_lines.toLocaleString()} 条 · ${d.size_mb} MB</div>
                <div class="date-bar" style="width:${pct}%"></div>
            </div>`;
        }).join('');
    } catch (e) { alert('加载失败: ' + e.message); }
    finally { hideLoading(); }
}

async function selectDate(date) {
    currentDate = date;
    showLoading(`正在加载 ${date} 的数据...`);
    datePanel.style.display = 'none';
    dayPanel.style.display = 'block';
    backBtn.style.display = 'inline-block';
    document.getElementById('currentDate').textContent =
        `${date.slice(0,4)}-${date.slice(4,6)}-${date.slice(6,8)}`;
    try {
        await loadHourlyChart();
        initSlider();
        selectPreset('all');
        rawOffset = 0;
        await Promise.all([loadFilters(), loadRawSpans(), loadTraces()]);
        loadTimeline();
    } catch (e) { alert('加载失败: ' + e.message); }
    finally { hideLoading(); }
}

function backToDates() {
    datePanel.style.display = 'block';
    dayPanel.style.display = 'none';
    backBtn.style.display = 'none';
    currentDate = null;
}

/* ========== 24小时分布表格 ========== */
async function loadHourlyChart() {
    const res = await fetch(`/api/day_hourly?date=${currentDate}`);
    const data = await res.json();
    hourlyData = data.hours || [];
    renderHourlyChart();
}

function renderHourlyChart() {
    const tbody = document.querySelector('#hourlyTable tbody');
    if (!hourlyData.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px;">暂无数据</td></tr>';
        return;
    }
    const maxCount = Math.max(...hourlyData.map(d => d.count), 1);
    tbody.innerHTML = hourlyData.map(d => {
        const pct = (d.count / maxCount) * 100;
        const names = d.names.slice(0, 3).join(', ') || '-';
        return `
            <tr onclick="selectHour(${d.hour})" style="cursor:pointer;">
                <td><span style="font-weight:bold;font-family:monospace;">${d.hour.toString().padStart(2,'0')}:00 - ${d.hour.toString().padStart(2,'0')}:59</span></td>
                <td>
                    <div style="position:relative;height:18px;background:var(--bar-bg);border-radius:3px;overflow:hidden;min-width:80px;">
                        <div style="height:100%;background:var(--accent);opacity:0.6;width:${pct}%;"></div>
                        <span style="position:absolute;left:4px;top:50%;transform:translateY(-50%);font-size:11px;font-weight:bold;color:var(--text);">${d.count}</span>
                    </div>
                </td>
                <td>${d.total_ms.toFixed(2)} ms</td>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-dim);" title="${names}">${names}</td>
                <td><button style="padding:2px 8px;font-size:11px;" onclick="event.stopPropagation();selectHour(${d.hour})">选中</button></td>
            </tr>`;
    }).join('');
}

function selectHour(hour) {
    sliderStartMin = hour * 60;
    sliderEndMin = Math.min(1439, (hour + 1) * 60 - 1);
    updateSliderUI();
    syncInputFromSlider();
    refreshAll();
}

/* ========== 滑块 ========== */
function initSlider() {
    const slider = document.getElementById('dualSlider');
    slider.onmousedown = (e) => {
        const rect = slider.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        const min = Math.round(pct * 1440);
        const distLeft = Math.abs(min - sliderStartMin);
        const distRight = Math.abs(min - sliderEndMin);
        isDragging = distLeft < distRight ? 'left' : 'right';
        updateSliderFromMouse(e, rect);
    };
    document.onmousemove = (e) => {
        if (!isDragging) return;
        updateSliderFromMouse(e, document.getElementById('dualSlider').getBoundingClientRect());
    };
    document.onmouseup = () => {
        if (isDragging) {
            isDragging = null;
            syncInputFromSlider();
            refreshAll();
        }
    };
}

function updateSliderFromMouse(e, rect) {
    let pct = (e.clientX - rect.left) / rect.width;
    pct = Math.max(0, Math.min(1, pct));
    const min = Math.round(pct * 1440);
    if (isDragging === 'left') sliderStartMin = Math.min(min, sliderEndMin - 5);
    else sliderEndMin = Math.max(min, sliderStartMin + 5);
    updateSliderUI();
}

function updateSliderUI() {
    const total = 1440;
    const lp = (sliderStartMin / total) * 100;
    const rp = (sliderEndMin / total) * 100;
    document.getElementById('thumbLeft').style.left = lp + '%';
    document.getElementById('thumbRight').style.left = rp + '%';
    document.getElementById('sliderRange').style.left = lp + '%';
    document.getElementById('sliderRange').style.width = (rp - lp) + '%';
    document.getElementById('sliderStart').textContent = formatMin(sliderStartMin);
    document.getElementById('sliderEnd').textContent = formatMin(sliderEndMin);
    const range = sliderEndMin - sliderStartMin;
    document.getElementById('sliderRangeText').textContent = range >= 1439 ? '全天' : `${Math.floor(range/60)}时${range%60}分`;
}

function formatMin(min) {
    const h = Math.floor(min / 60);
    const m = min % 60;
    return `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}`;
}

function selectPreset(preset) {
    document.querySelectorAll('.slider-actions button').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('preset-' + preset);
    if (btn) btn.classList.add('active');

    switch(preset) {
        case 'last1h': return; // 已禁用
        case 'morning': sliderStartMin = 6*60; sliderEndMin = 12*60; break;
        case 'afternoon': sliderStartMin = 12*60; sliderEndMin = 18*60; break;
        case 'evening': sliderStartMin = 18*60; sliderEndMin = 24*60-1; break;
        case 'night': sliderStartMin = 0; sliderEndMin = 6*60; break;
        case 'all': sliderStartMin = 0; sliderEndMin = 1439; break;
    }
    updateSliderUI();
    syncInputFromSlider();
    refreshAll();
}

function applyInputTime() {
    const startVal = document.getElementById('inputStart').value;
    const endVal = document.getElementById('inputEnd').value;
    if (!startVal || !endVal) return;
    const [sh, sm] = startVal.split(':').map(Number);
    const [eh, em] = endVal.split(':').map(Number);
    sliderStartMin = sh * 60 + sm;
    sliderEndMin = eh * 60 + em;
    updateSliderUI();
    refreshAll();
}

function syncInputFromSlider() {
    document.getElementById('inputStart').value = formatMin(sliderStartMin);
    document.getElementById('inputEnd').value = formatMin(sliderEndMin);
}

/* ========== 过滤下拉 ========== */
async function loadFilters() {
    if (!currentDate) return;
    try {
        const url = `/api/filters?date=${currentDate}&start_min=${sliderStartMin}&end_min=${sliderEndMin}`;
        const res = await fetch(url);
        const data = await res.json();
        dropdownData = { name: data.names, tag: data.tags, room: data.rooms };
        populateDropdown('name', data.names);
        populateDropdown('tag', data.tags);
        populateDropdown('room', data.rooms);
    } catch (e) { console.error(e); }
}

function populateDropdown(type, items) {
    const el = document.getElementById(type + 'Dropdown');
    const input = document.getElementById(type + 'Filter');
    const currentVal = input.value.trim().toLowerCase();
    let displayItems = currentVal ? items.filter(i => i.toLowerCase().includes(currentVal)) : items;
    if (!displayItems.length) { el.innerHTML = '<div class="dropdown-item empty">无匹配候选</div>'; return; }
    el.innerHTML = displayItems.map(item => {
        const esc = item.replace(/'/g, "\\'").replace(/"/g, '\\"');
        return `<div class="dropdown-item" onmousedown="selectFilter('${type}', '${esc}')">${item}</div>`;
    }).join('');
}

function selectFilter(type, value) {
    document.getElementById(type + 'Filter').value = value;
    hideDropdown(type);
    refreshAll();
}

function showDropdown(type) {
    const el = document.getElementById(type + 'Dropdown');
    populateDropdown(type, dropdownData[type] || []);
    el.style.display = 'block';
}
function hideDropdown(type) { document.getElementById(type + 'Dropdown').style.display = 'none'; }

['name', 'tag', 'room'].forEach(type => {
    const input = document.getElementById(type + 'Filter');
    input.addEventListener('focus', () => showDropdown(type));
    input.addEventListener('input', () => {
        populateDropdown(type, dropdownData[type] || []);
        if (document.getElementById(type + 'Dropdown').style.display !== 'block') showDropdown(type);
    });
});
document.addEventListener('click', (e) => {
    ['name', 'tag', 'room'].forEach(type => {
        const wrap = document.getElementById(type + 'Filter')?.closest('.filter-wrap');
        if (wrap && !wrap.contains(e.target)) hideDropdown(type);
    });
});

/* ========== 统计表格 ========== */
async function refreshStats() {
    if (!currentDate) return;
    showLoading('正在统计...');
    try {
        const n = document.getElementById('nameFilter').value.trim();
        const t = document.getElementById('tagFilter').value.trim();
        const r = document.getElementById('roomFilter').value.trim();
        let url = `/api/stats?date=${currentDate}&start_min=${sliderStartMin}&end_min=${sliderEndMin}`;
        if (n) url += '&name=' + encodeURIComponent(n);
        if (t) url += '&tag=' + encodeURIComponent(t);
        if (r) url += '&room=' + encodeURIComponent(r);

        const res = await fetch(url);
        const data = await res.json();
        const tbody = document.querySelector('#statsTable tbody');
        tbody.innerHTML = '';
        data.forEach(s => {
            const cls = s.p95 > 50 ? 'slow' : s.p95 > 30 ? 'warn' : 'ok';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td title="${s.name}">${s.name.length > 50 ? s.name.slice(0,47)+'...' : s.name}</td>
                <td>${s.cnt}</td><td class="${cls}">${s.avg_ms}ms</td>
                <td>${s.min_ms}</td><td>${s.max_ms}</td>
                <td>${s.p50}ms</td><td class="${cls}">${s.p95}ms</td>
                <td>${s.p99}ms</td><td>${s.total_ms}ms</td><td>${s.pct}%</td>`;
            tbody.appendChild(tr);
        });
    } catch (e) { console.error(e); }
    finally { hideLoading(); }
}

/* ========== Trace 树 ========== */
async function loadTraces() {
    if (!currentDate) return;
    try {
        const n = document.getElementById('nameFilter').value.trim();
        const t = document.getElementById('tagFilter').value.trim();
        const r = document.getElementById('roomFilter').value.trim();
        let url = `/api/traces?date=${currentDate}&start_min=${sliderStartMin}&end_min=${sliderEndMin}&limit=0`;
        if (n) url += '&name=' + encodeURIComponent(n);
        if (t) url += '&tag=' + encodeURIComponent(t);
        if (r) url += '&room=' + encodeURIComponent(r);

        const res = await fetch(url);
        const traces = await res.json();
        const container = document.getElementById('traceList');
        if (!traces.length) {
            container.innerHTML = '<div class="loading-inline">当前时间段暂无 Trace 数据</div>';
            return;
        }
        container.innerHTML = traces.map(t => {
            const tags = t.tags && t.tags.length ? ` [${t.tags.join(',')}]` : '';
            const startTime = formatLocal(t.start_us);
            const matchCls = t.matches_filter ? 'trace-match' : '';
            return `
            <div class="trace-item ${matchCls}" onclick="showTraceTree('${t.trace}')">
                <div>
                    <span class="trace-name">${t.name}</span>
                    <span class="trace-meta">${tags}</span>
                </div>
                <div class="trace-meta">${startTime} · ${t.ms}ms · ${t.children_count} 子节点 · ${t.uid || '-'} · ${t.room || '-'}</div>
            </div>`;
        }).join('');
    } catch (e) {
        console.error(e);
        document.getElementById('traceList').innerHTML = '<div class="loading-inline">加载失败</div>';
    }
}

async function showTraceTree(trace) {
    if (!currentDate || !trace) return;
    showLoading('正在加载 Trace 树...');
    try {
        const n = document.getElementById('nameFilter').value.trim();
        const t = document.getElementById('tagFilter').value.trim();
        const r = document.getElementById('roomFilter').value.trim();
        let url = `/api/trace_tree?date=${currentDate}&root_trace=${trace}`;
        if (n) url += '&name=' + encodeURIComponent(n);
        if (t) url += '&tag=' + encodeURIComponent(t);
        if (r) url += '&room=' + encodeURIComponent(r);

        const res = await fetch(url);
        const data = await res.json();
        if (!data.tree || !data.tree.name) { alert('Trace 数据为空'); return; }
        document.getElementById('traceTreeContainer').style.display = 'block';
        const rootTime = formatLocal(data.tree.start_us);
        document.getElementById('treeRootName').textContent =
            `${data.tree.name} (${data.tree.ms}ms) [${data.root_trace}] @ ${rootTime}`;
        const treeEl = document.getElementById('traceTree');
        treeEl.innerHTML = renderTreeNode(data.tree, 0, null);
        document.getElementById('traceTreeContainer').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) { console.error(e); alert('加载 Trace 树失败'); }
    finally { hideLoading(); }
}

function escAttr(s) {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function msColorClass(ms) {
    if (ms > 30) return 'ms-slow';
    if (ms > 10) return 'ms-warn';
    return 'ms-fast';
}

function renderTreeNode(node, depth, parentMs) {
    const hasChildren = node.children && node.children.length > 0;
    const tags = node.tags && node.tags.length ? ` [${node.tags.join(',')}]` : '';
    const startTime = formatLocal(node.start_us);
    const traceId = escAttr(node.trace);
    const msCls = msColorClass(node.ms);
    const matches = node.matches_filter;
    const inPath = node.in_match_path;
    let matchCls = '';
    if (matches) matchCls = 'tree-match';
    else if (!inPath) matchCls = 'tree-dim';

    let ratioHtml = '';
    if (parentMs && parentMs > 0) {
        const ratio = Math.round((node.ms / parentMs) * 100);
        ratioHtml = `<span class="tree-ratio">${ratio}%</span>`;
    }
    let exclusiveHtml = '';
    if (hasChildren && node.children_ms_sum !== undefined) {
        const exclusive = Math.round((node.ms - node.children_ms_sum) * 100) / 100;
        if (exclusive > 0) exclusiveHtml = `<span class="tree-node-exclusive positive">+${exclusive}ms</span>`;
    }
    const childrenHtml = hasChildren ?
        `<div class="tree-children" id="tree-children-${traceId}">${node.children.map(c => renderTreeNode(c, depth + 1, node.ms)).join('')}</div>` : '';

    return `
        <div class="tree-node">
            <div class="tree-node-content ${matchCls}" onclick="toggleTreeNode('${traceId}', this)">
                <span class="tree-toggle">${hasChildren ? '▼' : ' '}</span>
                <span class="tree-node-time">${startTime}</span>
                <span class="tree-node-name" title="${escAttr(node.name)}">${node.name}</span>
                <span class="tree-node-ms ${msCls}">${node.ms}ms</span>
                ${ratioHtml}
                <span class="tree-node-meta">${tags} · ${node.uid || '-'} · ${node.room || '-'}</span>
                ${exclusiveHtml}
            </div>
            ${childrenHtml}
        </div>`;
}

function toggleTreeNode(trace, el) {
    const children = document.getElementById(`tree-children-${trace}`);
    const toggle = el.querySelector('.tree-toggle');
    if (children) {
        if (children.style.display === 'none') {
            children.style.display = 'block';
            toggle.textContent = '▼';
        } else {
            children.style.display = 'none';
            toggle.textContent = '▶';
        }
    }
}
function closeTree() { document.getElementById('traceTreeContainer').style.display = 'none'; }

/* ========== 原始记录分页 ========== */
function updatePageIndicator(total, offset, limit) {
    const totalPages = Math.ceil(total / limit) || 1;
    const currentPage = Math.floor(offset / limit) + 1;
    document.getElementById('pageIndicator').textContent = `第 ${currentPage} / ${totalPages} 页`;
    document.getElementById('jumpPage').value = currentPage;
    document.getElementById('jumpPage').max = totalPages;
    document.getElementById('firstBtn').disabled = currentPage <= 1;
    document.getElementById('prevBtn').disabled = currentPage <= 1;
    document.getElementById('nextBtn').disabled = offset + limit >= total;
    document.getElementById('lastBtn').disabled = offset + limit >= total;
}

async function loadRawSpans() {
    if (!currentDate) return;
    const size = parseInt(document.getElementById('pageSize').value) || 50;
    rawPageSize = size;
    const n = document.getElementById('nameFilter').value.trim();
    const t = document.getElementById('tagFilter').value.trim();
    const r = document.getElementById('roomFilter').value.trim();

    showLoading('正在加载原始记录...');
    try {
        let url = `/api/raw_spans?date=${currentDate}&offset=${rawOffset}&limit=${rawPageSize}&start_min=${sliderStartMin}&end_min=${sliderEndMin}`;
        if (n) url += '&name=' + encodeURIComponent(n);
        if (t) url += '&tag=' + encodeURIComponent(t);
        if (r) url += '&room=' + encodeURIComponent(r);

        const res = await fetch(url);
        const data = await res.json();
        const tbody = document.querySelector('#rawTable tbody');
        tbody.innerHTML = '';
        data.spans.forEach(s => {
            const ts = formatLocal(s.start_us);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${ts}</td>
                <td><span style="word-break:break-all;">${s.name}</span></td>
                <td>${s.ms}ms</td>
                <td>${s.uid || '-'}</td>
                <td>${s.room || '-'}</td>
                <td>${(s.tags||[]).join(', ')}</td>
                <td><code style="cursor:pointer;color:var(--accent);" onclick="showTraceTree('${escAttr(s.trace)}')" title="查看 Trace 树">${s.trace?.slice(0,8) || '-'}</code></td>`;
            tbody.appendChild(tr);
        });
        document.getElementById('rawPageInfo').textContent =
            `第 ${rawOffset + 1}-${Math.min(rawOffset + data.spans.length, data.total)} 条 / 共 ${data.total} 条`;
        updatePageIndicator(data.total, data.offset, data.limit);
    } catch (e) { console.error(e); }
    finally { hideLoading(); }
}

function changePage(dir) {
    const size = parseInt(document.getElementById('pageSize').value) || 50;
    if (dir === 'first') rawOffset = 0;
    else if (dir === 'last') {
        const infoText = document.getElementById('rawPageInfo').textContent;
        const totalMatch = infoText.match(/共 (\d+) 条/);
        const total = totalMatch ? parseInt(totalMatch[1]) : 0;
        rawOffset = Math.max(0, Math.ceil(total / size) * size - size);
    } else {
        rawOffset += dir * size;
        if (rawOffset < 0) rawOffset = 0;
    }
    loadRawSpans();
}
function jumpToPage() {
    const size = parseInt(document.getElementById('pageSize').value) || 50;
    const page = parseInt(document.getElementById('jumpPage').value) || 1;
    const maxPage = parseInt(document.getElementById('jumpPage').max) || 1;
    const targetPage = Math.max(1, Math.min(page, maxPage));
    rawOffset = (targetPage - 1) * size;
    loadRawSpans();
}

/* ========== Timeline 时序视图 ========== */
async function loadTimeline() {
    if (!currentDate) return;
    const n = document.getElementById('nameFilter').value.trim();
    const t = document.getElementById('tagFilter').value.trim();
    const r = document.getElementById('roomFilter').value.trim();
    let url = `/api/timeline?date=${currentDate}&start_min=${sliderStartMin}&end_min=${sliderEndMin}`;
    if (n) url += '&name=' + encodeURIComponent(n);
    if (t) url += '&tag=' + encodeURIComponent(t);
    if (r) url += '&room=' + encodeURIComponent(r);

    try {
        const res = await fetch(url);
        timelineData = await res.json();
        window._cachedLanes = null;
        window._cachedLaneKey = null;
        renderTimeline();
    } catch (e) { console.error('timeline load failed', e); }
}

function renderTimeline() {
    const canvas = document.getElementById('timelineCanvas');
    const wrap = document.getElementById('timelineWrap');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('timelineTooltip');

    if (!timelineData.length) {
        canvas.width = wrap.clientWidth || 800;
        canvas.height = 60;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--text-muted');
        ctx.textAlign = 'center';
        ctx.font = '13px monospace';
        ctx.fillText('当前时间段无数据', canvas.width / 2, 35);
        wrap.onwheel = null;
        return;
    }

    // 1. Swimlane 分配（缓存）
    const laneKey = timelineData.length + '_' + (timelineData[0]?.trace || '') + '_' + (timelineData[timelineData.length - 1]?.trace || '');
    if (!window._cachedLanes || window._cachedLaneKey !== laneKey) {
        const sorted = [...timelineData].sort((a, b) => a.start_us - b.start_us);
        const lanes = [];
        for (const span of sorted) {
            const spanEnd = span.start_us + Math.round(span.ms * 1000);
            let placed = false;
            for (const lane of lanes) {
                const last = lane.blocks[lane.blocks.length - 1];
                const lastEnd = last.start_us + Math.round(last.ms * 1000);
                if (span.start_us >= lastEnd) {
                    lane.blocks.push(span);
                    placed = true;
                    break;
                }
            }
            if (!placed) lanes.push({ blocks: [span] });
        }
        window._cachedLanes = lanes;
        window._cachedLaneKey = laneKey;
    }
    const lanes = window._cachedLanes;

    // 2. 尺寸计算（限制 Canvas 最大宽度）
    const padLeft = 80;
    const padRight = 10;
    const padTop = 24;
    const padBottom = 8;
    const laneHeight = 20;
    const MAX_CANVAS_WIDTH = 30000;

    const starts = timelineData.map(s => s.start_us);
    const ends = timelineData.map(s => s.start_us + Math.round(s.ms * 1000));
    const minT = Math.min(...starts);
    const maxT = Math.max(...ends);
    const range = Math.max(maxT - minT, 1);

    const containerW = wrap.clientWidth || 1200;
    const maxScale = (MAX_CANVAS_WIDTH - padLeft - padRight) / containerW;
    if (tlScale > maxScale) {
        tlScale = maxScale;
        updateTlZoomUI();
    }

    const drawWidth = Math.max(containerW * tlScale, containerW);
    const W = Math.floor(padLeft + drawWidth + padRight);
    const H = padTop + lanes.length * laneHeight + padBottom;

    canvas.width = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    // 3. 时间 → X
    const timeToX = (us) => padLeft + ((us - minT) / range) * drawWidth;

    // 4. 刻度（固定 10 个）
    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--border-light');
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--text-muted');
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    for (let i = 0; i <= 10; i++) {
        const t = minT + (range * i / 10);
        const x = timeToX(t);
        ctx.beginPath();
        ctx.moveTo(x, padTop - 4);
        ctx.lineTo(x, padTop);
        ctx.stroke();
        ctx.fillText(formatLocal(t), x, padTop - 6);
    }

    // 5. 绘制条块
    timelineLayout = [];

    for (let li = 0; li < lanes.length; li++) {
        const lane = lanes[li];
        const y = padTop + li * laneHeight;

        if (li % 2 === 1) {
            ctx.fillStyle = 'rgba(128,128,128,0.04)';
            ctx.fillRect(0, y, W, laneHeight);
        }

        for (const span of lane.blocks) {
            const x = timeToX(span.start_us);
            const w = Math.max(1, (span.ms * 1000 / range) * drawWidth);
            const color = nameToColor(span.name);

            ctx.fillStyle = color;
            ctx.globalAlpha = 0.85;
            ctx.fillRect(x, y + 1, w, laneHeight - 2);
            ctx.globalAlpha = 1;

            ctx.strokeStyle = 'rgba(0,0,0,0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y + 1, w, laneHeight - 2);

            if (w > 40) {
                ctx.fillStyle = '#000';
                ctx.font = '10px monospace';
                ctx.textAlign = 'left';
                const txt = span.name.length > 20 ? span.name.slice(0, 18) + '..' : span.name;
                ctx.fillText(txt, x + 3, y + 14);
            }

            timelineLayout.push({ x, y: y + 1, w, h: laneHeight - 2, span });
        }
    }

    // 6. 滚轮缩放
    wrap.onwheel = (e) => {
        e.preventDefault();
        const anchorOffsetX = e.offsetX;
        const anchorTime = minT + ((anchorOffsetX - padLeft) / drawWidth) * range;

        if (e.deltaY < 0) tlScale *= 1.2;
        else tlScale /= 1.2;
        tlScale = Math.max(0.01, tlScale);

        let newDrawWidth = Math.max(containerW * tlScale, containerW);
        if (padLeft + newDrawWidth + padRight > MAX_CANVAS_WIDTH) {
            newDrawWidth = MAX_CANVAS_WIDTH - padLeft - padRight;
        }
        const newAnchorOffsetX = padLeft + ((anchorTime - minT) / range) * newDrawWidth;
        wrap.scrollLeft += (newAnchorOffsetX - anchorOffsetX);

        updateTlZoomUI();
        renderTimeline();
    };

    // 7. 悬停 / 点击（Tooltip fixed 定位，显示完整函数名）
    canvas.onmousemove = (e) => {
        const mx = e.offsetX;
        const my = e.offsetY;
        const hit = timelineLayout.find(b => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);

        if (hit) {
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 12) + 'px';
            tooltip.style.top = (e.clientY - 10) + 'px';
            const s = hit.span;
            const tagsStr = s.tags && s.tags.length ? ` [${s.tags.join(',')}]` : '';
            const uidStr = s.uid ? ` uid=${s.uid}` : '';
            const roomStr = s.room ? ` room=${s.room}` : '';

            tooltip.innerHTML =
                `<div style="font-weight:bold;font-size:12px;color:#fff;margin-bottom:4px;">${s.name}</div>` +
                `<div style="color:var(--accent);">${s.ms}ms · ${formatLocal(s.start_us)}</div>` +
                `<div style="color:var(--text-muted);font-size:10px;margin-top:4px;">` +
                `trace: ${s.trace}${tagsStr}${uidStr}${roomStr}` +
                `</div>`;
            canvas.style.cursor = 'pointer';
        } else {
            tooltip.style.display = 'none';
            canvas.style.cursor = 'default';
        }
    };
    canvas.onmouseleave = () => { tooltip.style.display = 'none'; };
    canvas.onclick = (e) => {
        const mx = e.offsetX;
        const my = e.offsetY;
        const hit = timelineLayout.find(b => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);
        if (hit) showTraceTree(hit.span.trace);
    };

    updateTlZoomUI();
}

function nameToColor(name) {
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
    const hue = Math.abs(hash % 360);
    const sat = 65 + (Math.abs(hash >> 8) % 25);
    const lig = 50 + (Math.abs(hash >> 16) % 15);
    return `hsl(${hue}, ${sat}%, ${lig}%)`;
}

/* ========== 全局刷新 ========== */
async function refreshAll() {
    await loadFilters();
    refreshStats();
    loadRawSpans();
    loadTraces();
    loadTimeline();
}

let debounceTimer;
document.getElementById('nameFilter').addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { populateDropdown('name', dropdownData.name || []); refreshAll(); }, 400);
});
document.getElementById('tagFilter').addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { populateDropdown('tag', dropdownData.tag || []); refreshAll(); }, 400);
});
document.getElementById('roomFilter').addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { populateDropdown('room', dropdownData.room || []); refreshAll(); }, 400);
});
document.getElementById('pageSize').addEventListener('change', () => { rawOffset = 0; loadRawSpans(); });

/* ========== 启动 ========== */
initTheme();
loadDates();




