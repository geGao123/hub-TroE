// timiAgent Web — 调试友好的 chat UI
// 思路: 每个右侧面板独立 render 自己的数据, 左侧 chat 区域只是 "投影"

const $ = (id) => document.getElementById(id);

// === State ===
let sessionId = null;
let isStreaming = false;

// === Helper: pretty JSON (保留 unicode) ===
function fmtJSON(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch (e) {
    return String(obj);
  }
}

// JSON syntax highlight (返回 HTML)
function fmtJSONHL(obj) {
  const json = fmtJSON(obj);
  // escape HTML first
  const esc = json
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return esc.replace(
    /("(?:[^"\\]|\\.)*")(\s*:)?|(\b(?:true|false)\b)|(\bnull\b)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}\[\],])/g,
    (m, str, colon, bool, nul, num, punct) => {
      if (str !== undefined) {
        if (colon) return `<span class="j-key">${str}</span><span class="j-punct">${colon}</span>`;
        return `<span class="j-str">${str}</span>`;
      }
      if (bool) return `<span class="j-bool">${bool}</span>`;
      if (nul) return `<span class="j-null">${nul}</span>`;
      if (num) return `<span class="j-num">${num}</span>`;
      if (punct) return `<span class="j-punct">${punct}</span>`;
      return m;
    }
  );
}

function truncate(s, len = 80) {
  if (!s) return '';
  s = String(s);
  return s.length > len ? s.slice(0, len) + '…' : s;
}

function now() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

// =============================================================
//  左: 聊天区
// =============================================================

function addChatMsg(role, summary, fullData, callId = null) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.tabIndex = 0;

  if (callId) {
    // 给 chat 消息标上 call-id, 供 wire 面板的 call-header 点击跳转
    div.dataset.callId = callId;
  }

  const tag = document.createElement('span');
  tag.className = 'role-tag';
  tag.textContent = role;
  div.appendChild(tag);

  const sum = document.createElement('span');
  sum.className = 'summary';
  sum.textContent = summary;
  div.appendChild(sum);

  if (fullData !== undefined) {
    const pre = document.createElement('pre');
    pre.className = 'json-preview';
    pre.textContent = fmtJSON(fullData);
    div.appendChild(pre);
    div.addEventListener('click', () => div.classList.toggle('expanded'));
  }

  $('messages').appendChild(div);
  $('messages').scrollTop = $('messages').scrollHeight;
  updateStats();
  return div;
}

function updateStats() {
  const msgs = $('messages').querySelectorAll('.msg').length;
  $('stats').textContent = `${msgs} msgs`;
}

function clearChat() {
  $('messages').innerHTML = '';
  updateStats();
  // 清空右侧 Rounds 面板
  const panel = $('panel-wire');
  panel.innerHTML = '<div class="placeholder">发送一条消息开始, 故事线会按 Round 分组, 点 LLM 调用可展开查看请求/响应</div>';
  roundCounter = 0;
  currentRound = null;
  pendingRequest = null;
  currentCallEl = null;
}

// =============================================================
//  右: System Prompt 面板
// =============================================================

async function loadSystemPrompt() {
  try {
    const res = await fetch('/api/system');
    const { prompt } = await res.json();
    $('system-prompt').textContent = prompt;
  } catch (e) {
    $('system-prompt').textContent = `(加载失败: ${e.message})`;
  }
}

// =============================================================
//  右: Tool Schemas 面板
// =============================================================

async function loadTools() {
  const res = await fetch('/api/tools');
  const data = await res.json();
  const { tools, schemas } = data;
  $('tools-count').textContent = tools.length;

  const container = $('tools-detail');
  container.innerHTML = '';

  for (const t of tools) {
    const schema = schemas.find(s => s.function.name === t.name) || {
      type: "function",
      function: { name: t.name, description: t.description, parameters: {} }
    };

    const card = document.createElement('div');
    card.className = 'tool-card collapsed';
    card.innerHTML = `
      <div class="tool-card-head">
        <span class="tool-card-name">${t.name}</span>
        <span class="chevron">▼</span>
      </div>
      <div class="tool-card-desc">${t.description}</div>
      <pre class="tool-card-schema">${fmtJSON(schema)}</pre>
    `;
    card.querySelector('.tool-card-head').addEventListener('click', () => {
      card.classList.toggle('collapsed');
    });
    container.appendChild(card);
  }
}

// =============================================================
//  右: Rounds 面板 (故事线 + LLM 调用详情, 融合 history + wire)
// =============================================================

// 状态
let roundCounter = 0;
let currentRound = null;        // 当前 round 容器 (.round-body, story-item append 上去)
let pendingRequest = null;      // 等 response 配对的 raw_request
let currentCallEl = null;       // 当前正在构建的 story-item llm-call (tool_call/tool_result 会嵌进去)

// 给故事线添加一条 item
function addStoryItem(kind, content, callId = null, extra = {}) {
  if (!currentRound) return;
  const item = document.createElement('div');
  item.className = `story-item story-${kind}`;
  if (callId) item.dataset.callId = callId;

  const tag = document.createElement('span');
  tag.className = 'story-tag';
  tag.textContent = kind;
  item.appendChild(tag);

  const body = document.createElement('span');
  body.className = 'story-body';
  body.textContent = content;
  item.appendChild(body);

  if (extra.sub) {
    const sub = document.createElement('span');
    sub.className = 'story-sub';
    sub.textContent = extra.sub;
    item.appendChild(sub);
  }

  currentRound.appendChild(item);
  return item;
}

// 创建 "LLM call" 故事线条目 (请求 + 响应完后展开可看详情)
function createCallStoryItem(callId) {
  if (!currentRound) return;
  const item = document.createElement('div');
  item.className = 'story-item story-call';
  item.dataset.callId = callId;

  const tag = document.createElement('span');
  tag.className = 'story-tag';
  tag.textContent = 'LLM';
  item.appendChild(tag);

  const head = document.createElement('div');
  head.className = 'story-call-head';
  head.innerHTML = `
    <span class="story-call-toggle">▼</span>
    <span class="story-call-summary">…</span>
    <span class="story-call-meta"></span>
  `;
  item.appendChild(head);

  // 详情容器 (默认折叠, 点击 head 展开)
  const detail = document.createElement('div');
  detail.className = 'story-call-detail';
  detail.innerHTML = '<div class="story-call-empty">(等待响应…)</div>';
  item.appendChild(detail);

  // 点击 toggle 详情
  head.addEventListener('click', () => {
    item.classList.toggle('expanded');
    const tg = head.querySelector('.story-call-toggle');
    if (tg) tg.textContent = item.classList.contains('expanded') ? '▲' : '▼';
  });

  currentRound.appendChild(item);
  return item;
}

// 从 raw_request / raw_response 抽出简略摘要
function extractCallSummary(finishReason, response) {
  if (finishReason === 'tool_calls') {
    const tcs = response?.choices?.[0]?.message?.tool_calls || [];
    if (tcs.length === 0) return '(no tool_calls)';
    if (tcs.length === 1) {
      const tc = tcs[0];
      return `🔧 ${tc.function?.name || '?'}(${truncate(tc.function?.arguments || '{}', 50)})`;
    }
    const names = tcs.map(tc => tc.function?.name || '?').join(', ');
    return `🔧 ${names} (${tcs.length} 个)`;
  }
  if (finishReason === 'stop') {
    const text = response?.choices?.[0]?.message?.content || '';
    return `💬 ${truncate(text, 60)}`;
  }
  if (finishReason === 'length') return `⚠️ 截断 (length)`;
  return `(${finishReason})`;
}

function startRound(data) {
  roundCounter++;
  const panel = $('panel-wire');
  if (panel.querySelector('.placeholder')) panel.innerHTML = '';

  const round = document.createElement('div');
  round.className = 'round';
  if (data.round_id) round.dataset.roundId = data.round_id;

  // Round header (可折叠整轮)
  const header = document.createElement('div');
  header.className = 'round-header';
  header.innerHTML = `
    <span class="round-num">Round #${roundCounter}</span>
    <span class="round-user">"${truncate(data.user_text, 60)}"</span>
    <span class="round-stats"><span class="call-count">0</span> 次 LLM 调用 · msgs: ${data.messages_before}</span>
    <span class="round-toggle">▼</span>
  `;
  round.appendChild(header);

  // 故事线容器: user / llm-call / tool_call / tool_result / assistant 按时间顺序追加
  const body = document.createElement('div');
  body.className = 'round-body';
  round.appendChild(body);

  // 点击 header 折叠/展开整轮
  header.style.cursor = 'pointer';
  header.addEventListener('click', () => {
    round.classList.toggle('collapsed');
    const tg = header.querySelector('.round-toggle');
    if (tg) tg.textContent = round.classList.contains('collapsed') ? '▶' : '▼';
  });

  panel.appendChild(round);
  panel.scrollTop = panel.scrollHeight;

  // 故事线第 1 行: user
  addStoryItem('user', truncate(data.user_text, 80));

  currentRound = body;
  pendingRequest = null;
  currentCallEl = null;
}

function endRound(data) {
  if (!currentRound) return;
  // 更新 stats
  const round = currentRound.parentElement;
  const stats = round.querySelector('.round-stats');
  if (stats) {
    stats.innerHTML = `<span class="call-count">${stats.querySelector('.call-count')?.textContent || 0}</span> 次 LLM 调用 · msgs: ${data.messages_before} → ${data.messages_after}`;
  }
  currentRound = null;
  currentCallEl = null;
}

function addWireRequest(data) {
  pendingRequest = data;
}

function addWireResponse(data) {
  if (!pendingRequest || !currentRound) return;

  const callId = data.call_id || 'unknown';

  // 创建 LLM call story-item (请求已收到, 等待响应填充 detail)
  const callEl = createCallStoryItem(callId);
  currentCallEl = callEl;

  // 填充 detail
  const msgs = pendingRequest.request?.messages || [];
  const tools = pendingRequest.request?.tools || [];

  const usage = data.response?.usage || {};
  const usageStr = usage.total_tokens
    ? `${usage.total_tokens} tok (prompt=${usage.prompt_tokens}, comp=${usage.completion_tokens})`
    : '';

  // 头部 summary
  const summary = extractCallSummary(data.finish_reason, data.response);
  callEl.querySelector('.story-call-summary').textContent = summary;
  callEl.querySelector('.story-call-meta').textContent = `msgs=${msgs.length} · tools=${tools.length} · [${data.finish_reason}]${usageStr ? ' · ' + usageStr : ''}`;

  // detail 内容
  const detail = callEl.querySelector('.story-call-detail');
  detail.innerHTML = '';

  // REQUEST block
  const reqLabel = document.createElement('div');
  reqLabel.className = 'wire-block-label req';
  reqLabel.innerHTML = `<span class="arrow">▶</span> REQUEST <span class="wire-block-meta">POST /v1/chat/completions · ${msgs.length} msgs · ${tools.length} tools</span>`;
  const reqBody = document.createElement('pre');
  reqBody.className = 'wire-block-body';
  reqBody.innerHTML = fmtJSONHL(pendingRequest.request);
  detail.appendChild(reqLabel);
  detail.appendChild(reqBody);

  // RESPONSE block
  const resLabel = document.createElement('div');
  resLabel.className = 'wire-block-label res';
  resLabel.innerHTML = `<span class="arrow">◀</span> RESPONSE <span class="wire-block-meta">finish_reason=${data.finish_reason}${usageStr ? ' · ' + usageStr : ''}</span>`;
  const resBody = document.createElement('pre');
  resBody.className = 'wire-block-body res-body';
  resBody.innerHTML = fmtJSONHL(data.response);
  detail.appendChild(resLabel);
  detail.appendChild(resBody);

  // 默认折叠, 用户手动点 head 才展开 (避免与用户折叠状态冲突)

  // 自动滚到底
  $('panel-wire').scrollTop = $('panel-wire').scrollHeight;

  // 更新 round 计数
  const roundEl = currentRound.parentElement;
  const callCountEl = roundEl.querySelector('.call-count');
  if (callCountEl) {
    const n = parseInt(callCountEl.textContent || '0', 10) + 1;
    callCountEl.textContent = n;
  }

  pendingRequest = null;
}

// =============================================================
//  发送消息 (主流程)
// =============================================================

async function sendMessage(text) {
  if (isStreaming) {
    console.warn('正在生成中, 请稍候');
    return;
  }
  if (!text.trim()) return;

  isStreaming = true;
  addChatMsg('user', text);

  const debugLevel = parseInt($('debug-level').value, 10);

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message: text,
      debug_level: debugLevel,
    }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        if (!part.trim()) continue;
        const lines = part.split('\n');
        let evtName = 'message';
        let evtData = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) evtName = line.slice(7).trim();
          else if (line.startsWith('data: ')) evtData = line.slice(6);
        }
        if (evtData) {
          try {
            const data = JSON.parse(evtData);
            handleEvent(evtName, data);
          } catch (e) {
            console.warn('SSE parse failed:', e, evtData);
          }
        }
      }
    }
  } catch (e) {
    console.error('SSE stream error:', e);
    addChatMsg('error', `流读取失败: ${e.message}`);
  } finally {
    isStreaming = false;
  }
}

function handleEvent(evtName, data) {
  switch (evtName) {
    case 'session':
      sessionId = data.session_id;
      break;
    case 'user':
      // user_text 已经由 round_start 处理过了
      break;
    case 'tool_call':
      addChatMsg('tool-call', `🔧 ${data.name}(${data.arguments})`, data, data.call_id);
      addStoryItem('tool-call', `${data.name}(${truncate(data.arguments, 50)})`, data.call_id);
      break;
    case 'tool_result':
      addChatMsg('tool-result', `← ${data.output}`, data, data.call_id);
      addStoryItem('tool-result', `← ${truncate(data.output, 60)}`, data.call_id);
      break;
    case 'assistant':
      addChatMsg('assistant', data.text, { text: data.text }, data.call_id);
      addStoryItem('assistant', truncate(data.text, 80), data.call_id);
      break;
    case 'error':
      addChatMsg('error', data.message, data, data.call_id);
      addStoryItem('error', data.message, data.call_id);
      break;
    case 'raw_request':
      addWireRequest(data);
      break;
    case 'raw_response':
      addWireResponse(data);
      break;
    case 'round_start':
      startRound(data);
      break;
    case 'round_end':
      endRound(data);
      break;
    case 'done':
      break;
  }
}

// =============================================================
//  命令处理
// =============================================================

const COMMANDS = {
  '/clear': async () => {
    if (sessionId) {
      await fetch('/api/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
    }
    clearChat();
    addChatMsg('assistant', '🧹 已清空会话');
  },
  '/history': () => {
    const msgs = $('messages').querySelectorAll('.msg');
    if (msgs.length === 0) {
      addChatMsg('assistant', '📊 0 msgs');
      return;
    }
    const lines = [];
    msgs.forEach((m, i) => {
      const role = m.className.replace('msg ', '').split(' ')[0];
      const summary = m.querySelector('.summary')?.textContent || '';
      lines.push(`  ${i + 1}. [${role}] ${truncate(summary, 80)}`);
    });
    addChatMsg('assistant', `📊 ${msgs.length} msgs\n${lines.join('\n')}`);
  },
  '/tools': () => {
    const tools = document.querySelectorAll('#tools-detail .tool-card');
    const list = [...tools].map(c => {
      const name = c.querySelector('.tool-card-name').textContent;
      const desc = c.querySelector('.tool-card-desc').textContent;
      return `  ${name}: ${desc}`;
    }).join('\n');
    addChatMsg('assistant', `🛠 ${tools.length} 个工具:\n${list}`);
  },
  '/help': () => {
    addChatMsg('assistant', [
      '可用命令:',
      '  /clear    清空会话',
      '  /history  查看消息列表',
      '  /tools    列出可用工具',
      '  /help     打印本帮助',
    ].join('\n'));
  },
};

async function handleCommand(text) {
  const cmd = text.trim().toLowerCase();
  if (cmd in COMMANDS) {
    await COMMANDS[cmd]();
    return true;
  }
  return false;
}

// =============================================================
//  初始化
// =============================================================

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = $('input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  if (text.startsWith('/')) {
    const handled = await handleCommand(text);
    if (!handled) {
      addChatMsg('error', `未知命令: ${text} (试试 /help)`);
    }
    return;
  }

  sendMessage(text);
});

$('btn-clear').addEventListener('click', () => handleCommand('/clear'));

// 面板折叠 (点击 header 空白区域, 但不触发按钮)
document.querySelectorAll('.panel-header').forEach(h => {
  h.addEventListener('click', (e) => {
    // 按钮点击不触发折叠
    if (e.target.closest('.panel-btn')) return;
    const panel = h.parentElement;
    panel.classList.toggle('collapsed');
    const toggle = h.querySelector('.collapse-btn');
    if (toggle) toggle.textContent = panel.classList.contains('collapsed') ? '+' : '−';
  });
});

// 折叠按钮
document.querySelectorAll('.collapse-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const panel = btn.closest('.panel');
    panel.classList.toggle('collapsed');
    btn.textContent = panel.classList.contains('collapsed') ? '+' : '−';
  });
});

// 全屏模式: 一次只显示一个面板
let focusedPanel = null;

function enterFocus(panel) {
  if (focusedPanel) return;
  focusedPanel = panel;
  panel.classList.add('focused');
  $('debug-aside').classList.add('focus-mode');
  panel.querySelector('.focus-btn').classList.add('active');
  panel.querySelector('.focus-btn').textContent = '⤬';
  $('btn-exit-focus').style.display = 'inline-block';
}

function exitFocus() {
  if (!focusedPanel) return;
  focusedPanel.classList.remove('focused');
  focusedPanel.querySelector('.focus-btn').classList.remove('active');
  focusedPanel.querySelector('.focus-btn').textContent = '⛶';
  $('debug-aside').classList.remove('focus-mode');
  focusedPanel = null;
  $('btn-exit-focus').style.display = 'none';
}

document.querySelectorAll('.focus-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const panel = btn.closest('.panel');
    if (focusedPanel === panel) {
      exitFocus();
    } else {
      if (focusedPanel) exitFocus();
      enterFocus(panel);
    }
  });
});

$('btn-exit-focus').addEventListener('click', exitFocus);

// ESC 退出全屏
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && focusedPanel) {
    exitFocus();
  }
});

loadSystemPrompt();
loadTools();
addChatMsg('assistant', '你好！我是 timiAgent。试试问 "北京天气如何" 或 "现在几点 + 3 加 5"。\n打个 /help 看所有命令。');
