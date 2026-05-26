let allFiles = [];
let fileTree = null;
let currentPath = null;
let currentData = null;
let showRaw = false;

/* ==================== 初始化 ==================== */

async function loadFiles() {
    const treeEl = document.getElementById("file-tree");
    treeEl.innerHTML = '<div class="empty-state">加载中...</div>';

    try {
        const res = await fetch("/api/files");
        if (!res.ok) throw new Error("HTTP " + res.status);
        const json = await res.json();
        if (json.error) throw new Error(json.error);

        allFiles = json.files || [];
        fileTree = buildTree(allFiles);
        renderTree(treeEl, fileTree);
    } catch (e) {
        let msg = e.message || String(e);
        if (e.name === "TypeError" && msg.indexOf("fetch") !== -1) {
            msg = "无法连接到后端，请确认：\n1. Flask 已启动（python app.py）\n2. 访问地址是 http://localhost:5000/ \n3. 不要直接双击打开 HTML 文件";
        }
        treeEl.innerHTML = '<div class="empty-state" style="color:#ff7b72;white-space:pre-wrap;padding:20px">' + escapeHtml(msg) + '</div>';
    }
}

/* ==================== 文件树 ==================== */

function buildTree(files) {
    const root = { name: "", type: "folder", children: {}, expanded: true };
    for (let i = 0; i < files.length; i++) {
        const f = files[i];
        const parts = f.path.split("/");
        let node = root;
        for (let j = 0; j < parts.length; j++) {
            const part = parts[j];
            const isFile = j === parts.length - 1;
            if (!node.children[part]) {
                node.children[part] = isFile
                    ? { name: part, type: "file", path: f.path, size: f.size, data: f }
                    : { name: part, type: "folder", children: {}, expanded: false };
            }
            node = node.children[part];
        }
    }
    return root;
}

function renderTree(container, tree, searchTerm) {
    if (!allFiles.length) {
        container.innerHTML = '<div class="empty-state">暂无 .msgpack / .json 文件</div>';
        return;
    }
    const ul = document.createElement("div");
    renderNode(ul, tree, 0, searchTerm);
    container.innerHTML = "";
    container.appendChild(ul);
}

function renderNode(container, node, depth, searchTerm) {
    if (node.type === "file") {
        const matches = !searchTerm || node.name.toLowerCase().indexOf(searchTerm) !== -1 || node.path.toLowerCase().indexOf(searchTerm) !== -1;
        if (!matches) return;

        const el = document.createElement("div");
        el.className = "tree-file" + (node.path === currentPath ? " active" : "");
        el.style.setProperty("--depth", depth);
        el.innerHTML = '<span class="tree-chevron hidden">▾</span><span class="tree-icon">📄</span><span class="tree-label">' + escapeHtml(node.name) + '</span>';
        el.onclick = function() { selectFile(node.path); };
        container.appendChild(el);
        return;
    }

    const childrenArr = Object.values(node.children);
    const hasVisibleChildren = !searchTerm || childrenArr.some(function(c) {
        return c.type === "file" && (c.name.toLowerCase().indexOf(searchTerm) !== -1 || c.path.toLowerCase().indexOf(searchTerm) !== -1);
    }) || childrenArr.some(function(c) { return c.type === "folder"; });

    if (node.name && !hasVisibleChildren) return;

    if (node.name) {
        const folderEl = document.createElement("div");
        folderEl.className = "tree-folder";
        folderEl.style.setProperty("--depth", depth);
        const chevronClass = node.expanded ? "" : " collapsed";
        folderEl.innerHTML = '<span class="tree-chevron' + chevronClass + '">▾</span><span class="tree-icon">📁</span><span class="tree-label">' + escapeHtml(node.name) + '</span>';
        folderEl.onclick = function(e) {
            e.stopPropagation();
            node.expanded = !node.expanded;
            onSearch();
        };
        container.appendChild(folderEl);
    }

    if (node.expanded || searchTerm) {
        const childContainer = document.createElement("div");
        childContainer.className = "tree-children" + (node.expanded || searchTerm ? "" : " collapsed");
        for (let i = 0; i < childrenArr.length; i++) {
            renderNode(childContainer, childrenArr[i], depth + 1, searchTerm);
        }
        if (childContainer.children.length > 0) {
            container.appendChild(childContainer);
        }
    }
}

function onSearch() {
    const q = document.getElementById("search").value.trim().toLowerCase();
    const treeEl = document.getElementById("file-tree");
    if (!fileTree) return;
    if (q) expandAll(fileTree);
    renderTree(treeEl, fileTree, q || null);
}

function expandAll(node) {
    if (node.type === "folder") {
        node.expanded = true;
        const children = Object.values(node.children);
        for (let i = 0; i < children.length; i++) {
            expandAll(children[i]);
        }
    }
}

/* ==================== 文件选择与查看 ==================== */

async function selectFile(path) {
    currentPath = path;
    onSearch();

    const header = document.getElementById("chat-header");
    const title = document.getElementById("chat-title");
    const dl = document.getElementById("download-btn");
    const msgs = document.getElementById("chat-messages");

    header.style.display = "flex";
    title.textContent = path;
    dl.href = "/api/download?path=" + encodeURIComponent(path);
    msgs.innerHTML = '<div class="empty-state">解析中...</div>';
    showRaw = false;
    updateViewMode();

    try {
        const res = await fetch("/api/view?path=" + encodeURIComponent(path));
        const json = await res.json();
        if (json.error) {
            msgs.innerHTML = '<div class="empty-state" style="color:#ff7b72">' + escapeHtml(json.error) + '</div>';
            currentData = null;
            return;
        }
        currentData = json.data;
        renderChat(currentData, json.name, json.size);
    } catch (e) {
        msgs.innerHTML = '<div class="empty-state" style="color:#ff7b72">请求失败: ' + escapeHtml(e.message) + '</div>';
        currentData = null;
    }
}

/* ==================== 聊天渲染 ==================== */

function renderChat(data, filename, filesize) {
    const container = document.getElementById("chat-messages");
    const rawPre = document.getElementById("raw-content");

    rawPre.textContent = JSON.stringify(data, null, 2);

    const messages = parseMessages(data);

    if (!messages || !messages.length) {
        // 非对话格式，显示调试信息 + 原始 JSON
        container.innerHTML = '';
        const debugDiv = document.createElement("div");
        debugDiv.className = "message system";
        debugDiv.innerHTML = '<div class="avatar">⚙️</div><div class="bubble">' +
            '<strong>未识别为对话格式</strong><br><br>' +
            '检测到的数据类型: <code>' + escapeHtml(getDataType(data)) + '</code><br>' +
            '顶层键: <code>' + escapeHtml(getTopKeys(data)) + '</code><br><br>' +
            '支持的格式：<br>' +
            '1. <code>[{role, content}, ...]</code><br>' +
            '2. <code>{messages: [...]}</code><br>' +
            '3. <code>{conversation/dialog/turns/history: [...]}</code><br><br>' +
            '已显示原始数据 ↓' +
        '</div></div>';
        container.appendChild(debugDiv);

        const rawDiv = document.createElement("div");
        rawDiv.className = "message raw";
        rawDiv.innerHTML = '<div class="avatar">📄</div><div class="bubble"><pre>' + escapeHtml(JSON.stringify(data, null, 2)) + '</pre></div>';
        container.appendChild(rawDiv);
        return;
    }

    container.innerHTML = '';
    for (let i = 0; i < messages.length; i++) {
        const m = messages[i];
        const role = m.role || "unknown";
        const content = formatContent(m.content);
        const meta = m.timestamp || m.name || m.turn || ("#" + (i + 1));

        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + role;

        let avatarIcon = "🤖";
        let nameLabel = role;
        if (role === "user") { avatarIcon = "👤"; nameLabel = "用户"; }
        else if (role === "system") { avatarIcon = "⚙️"; nameLabel = "系统"; }
        else if (role === "assistant") { avatarIcon = "🤖"; nameLabel = "助手"; }

        msgDiv.innerHTML =
            '<div class="avatar" title="' + escapeHtml(role) + '">' + avatarIcon + '</div>' +
            '<div>' +
                '<div class="bubble">' + content + '</div>' +
                '<div class="msg-meta">' + escapeHtml(String(nameLabel)) + ' · ' + escapeHtml(String(meta)) + '</div>' +
            '</div>';

        container.appendChild(msgDiv);
    }

    container.scrollTop = container.scrollHeight;
}

/* ==================== 消息解析（更鲁棒） ==================== */

function parseMessages(data) {
    // 1. 直接数组
    if (Array.isArray(data) && data.length > 0) {
        if (isMessageArray(data)) return data;
    }

    // 2. 对象包装
    if (data && typeof data === "object" && !Array.isArray(data)) {
        const keys = ["messages", "conversation", "dialog", "turns", "history", "data", "items", "records"];
        for (let i = 0; i < keys.length; i++) {
            const arr = data[keys[i]];
            if (Array.isArray(arr) && arr.length > 0 && isMessageArray(arr)) {
                return arr;
            }
        }
        // 递归查找一层嵌套
        const allVals = Object.values(data);
        for (let i = 0; i < allVals.length; i++) {
            const v = allVals[i];
            if (Array.isArray(v) && v.length > 0 && isMessageArray(v)) {
                return v;
            }
        }
    }

    return null;
}

function isMessageArray(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return false;
    // 检查前3个元素，只要有1个像消息就算
    const checkCount = Math.min(arr.length, 3);
    for (let i = 0; i < checkCount; i++) {
        const item = arr[i];
        if (item && typeof item === "object") {
            const keys = Object.keys(item).map(function(k) { return k.toLowerCase(); });
            // 必须有 role 或 from/speaker/type 之一
            const hasRole = keys.indexOf("role") !== -1 || keys.indexOf("from") !== -1 ||
                           keys.indexOf("speaker") !== -1 || keys.indexOf("type") !== -1 ||
                           keys.indexOf("name") !== -1;
            // 必须有 content 或 message/text/value 之一
            const hasContent = keys.indexOf("content") !== -1 || keys.indexOf("message") !== -1 ||
                                keys.indexOf("text") !== -1 || keys.indexOf("value") !== -1 ||
                                keys.indexOf("body") !== -1 || keys.indexOf("data") !== -1;
            if (hasRole || hasContent) return true;
        }
    }
    return false;
}

function getDataType(data) {
    if (data === null) return "null";
    if (Array.isArray(data)) return "array[" + data.length + "]";
    if (typeof data === "object") return "object{" + Object.keys(data).length + " keys}";
    return typeof data;
}

function getTopKeys(data) {
    if (data && typeof data === "object") {
        const keys = Object.keys(data).slice(0, 10);
        return keys.join(", ") + (Object.keys(data).length > 10 ? "..." : "");
    }
    return "N/A";
}

/* ==================== 内容格式化 ==================== */

function formatContent(content) {
    if (content === null || content === undefined) return "<em style='color:#8b949e'>（空）</em>";

    // 字符串
    if (typeof content === "string") {
        let html = escapeHtml(content);
        html = html.replace(/```([\s\S]*?)```/g, function(_, code) {
            return '<pre>' + escapeHtml(code.replace(/^\n|\n$/g, "")) + '</pre>';
        });
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    // 数组（多模态）
    if (Array.isArray(content)) {
        let html = "";
        const attachments = [];
        for (let i = 0; i < content.length; i++) {
            const item = content[i];
            if (typeof item === "string") {
                html += escapeHtml(item) + " ";
            } else if (item && typeof item === "object") {
                const type = (item.type || "").toLowerCase();
                if (type === "text" && item.text) {
                    html += escapeHtml(String(item.text)) + " ";
                } else if (type === "image" || item.image || item.image_url) {
                    attachments.push("🖼️ 图片");
                } else if (type === "audio" || item.audio) {
                    attachments.push("🔊 音频");
                } else if (type === "video" || item.video) {
                    attachments.push("🎬 视频");
                } else {
                    html += '<pre style="margin:4px 0">' + escapeHtml(JSON.stringify(item, null, 2)) + '</pre>';
                }
            } else {
                html += escapeHtml(String(item)) + " ";
            }
        }
        if (attachments.length) {
            html += '<div class="msg-attachments">' + attachments.map(function(a) {
                return '<span class="attachment">' + a + '</span>';
            }).join("") + '</div>';
        }
        return html || "<em style='color:#8b949e'>（空内容）</em>";
    }

    // 对象（单条消息对象本身）
    if (typeof content === "object") {
        // 尝试提取常见字段
        const text = content.text || content.message || content.body || content.value || content.data;
        if (typeof text === "string") return formatContent(text);
        return '<pre>' + escapeHtml(JSON.stringify(content, null, 2)) + '</pre>';
    }

    return escapeHtml(String(content));
}

/* ==================== 视图切换 ==================== */

function toggleRaw() {
    showRaw = !showRaw;
    updateViewMode();
}

function updateViewMode() {
    const msgs = document.getElementById("chat-messages");
    const raw = document.getElementById("chat-raw");
    const btn = document.querySelector('.chat-actions .btn:last-child');
    if (showRaw) {
        msgs.style.display = "none";
        raw.style.display = "block";
        if (btn) btn.classList.add("active");
    } else {
        msgs.style.display = "flex";
        raw.style.display = "none";
        if (btn) btn.classList.remove("active");
    }
}

/* ==================== 工具函数 ==================== */

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

/* ==================== 启动 ==================== */

loadFiles();
