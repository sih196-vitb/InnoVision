/**
 * Central Command Dashboard - Tactical Frontend Controller.
 * Handles WebSocket alert feeds, Web Audio tactical sound alerts,
 * interactive virtual fence polygon drawing, and hardware telemetry polling.
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const liveClock = document.getElementById('live-clock');
  const muteAudioBtn = document.getElementById('mute-audio-btn');
  const streamFps = document.getElementById('stream-fps');
  const streamTracks = document.getElementById('stream-tracks');
  const zerodceBadge = document.getElementById('zerodce-badge');
  const zerodceStatus = document.getElementById('zerodce-status');
  
  // Video & Canvas
  const videoWrapper = document.getElementById('video-wrapper');
  const cctvFeed = document.getElementById('cctv-feed');
  const fenceCanvas = document.getElementById('fence-canvas');
  const toggleDrawBtn = document.getElementById('toggle-draw-btn');
  const cancelDrawBtn = document.getElementById('cancel-draw-btn');
  const saveDrawBtn = document.getElementById('save-draw-btn');
  const drawInstruction = document.getElementById('draw-instruction');
  const fenceTagsContainer = document.getElementById('fence-tags-container');

  // Graceful video feed error handling and auto-retry without flooding
  let feedRetryTimer = null;
  if (cctvFeed) {
    cctvFeed.addEventListener('error', () => {
      if (!feedRetryTimer) {
        console.warn('[Feed] Video stream disconnected. Retrying in 2.5s...');
        feedRetryTimer = setTimeout(() => {
          cctvFeed.src = '/video_feed?t=' + Date.now();
          feedRetryTimer = null;
        }, 2500);
      }
    });
  }

  // Mode Switcher Elements
  const modeManualBtn = document.getElementById('mode-manual-btn');
  const modeAutoBtn = document.getElementById('mode-auto-btn');
  const rescanAutoBtn = document.getElementById('rescan-auto-btn');
  let currentPerimeterMode = 'manual';

  // Alerts
  const alertFeed = document.getElementById('alert-feed-container');
  const noAlertsMsg = document.getElementById('no-alerts-msg');
  const clearAlertsBtn = document.getElementById('clear-alerts-btn');

  // Telemetry
  const gpuNameText = document.getElementById('gpu-name-text');
  const vramValText = document.getElementById('vram-val-text');
  const vramProgress = document.getElementById('vram-progress');
  const cpuPercentText = document.getElementById('cpu-percent-text');
  const cpuProgress = document.getElementById('cpu-progress');
  const ramValText = document.getElementById('ram-val-text');

  // Modal
  const fenceModal = document.getElementById('fence-modal');
  const fenceNameInput = document.getElementById('fence-name-input');
  const fenceTypeSelect = document.getElementById('fence-type-select');
  const fenceLoiterInput = document.getElementById('fence-loiter-input');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const modalCancelBtn = document.getElementById('modal-cancel-btn');
  const modalConfirmBtn = document.getElementById('modal-confirm-btn');

  // --- Audio Siren State (Synthesized Web Audio API) ---
  let audioMuted = false;
  let audioCtx = null;

  function playAlertChime(threatLevel) {
    if (audioMuted) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      const now = audioCtx.currentTime;
      if (threatLevel === 'CRITICAL') {
        // High-pitched warning two-tone chirp
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.setValueAtTime(1100, now + 0.1);
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
        osc.start(now);
        osc.stop(now + 0.35);
      } else {
        // Soft tactical radar blip
        osc.type = 'sine';
        osc.frequency.setValueAtTime(640, now);
        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
        osc.start(now);
        osc.stop(now + 0.2);
      }
    } catch (e) {
      console.warn("Audio synthesis:", e);
    }
  }

  muteAudioBtn.addEventListener('click', () => {
    audioMuted = !audioMuted;
    if (audioMuted) {
      muteAudioBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> MUTED';
      muteAudioBtn.classList.add('btn-danger');
    } else {
      muteAudioBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> AUDIO ON';
      muteAudioBtn.classList.remove('btn-danger');
    }
  });

  // --- Live Clock ---
  setInterval(() => {
    const d = new Date();
    liveClock.textContent = d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  }, 1000);

  // --- WebSocket Setup ---
  const socket = io();

  socket.on('connect', () => {
    console.log('[WebSocket] Connected to Central Command Server.');
  });

  socket.on('alert_event', (alert) => {
    renderAlertCard(alert);
    playAlertChime(alert.threat_level);
  });

  socket.on('fences_updated', () => {
    loadFences();
  });

  // --- Alert Feed Renderer ---
  function renderAlertCard(alert) {
    if (noAlertsMsg) {
      noAlertsMsg.classList.add('hidden');
    }

    const card = document.createElement('div');
    card.className = `alert-card ${alert.threat_level || 'HIGH'}`;

    const badgeClass = alert.threat_level === 'CRITICAL' ? 'badge-critical' :
                       alert.threat_level === 'INFO' ? 'badge-info' : 'badge-high';

    const thumbHtml = alert.crop_base64
      ? `<img class="alert-thumb" src="data:image/jpeg;base64,${alert.crop_base64}" alt="Offender Snapshot">`
      : `<div class="alert-thumb" style="display:flex;align-items:center;justify-content:center;color:#475569;"><i class="fa-solid fa-crosshairs"></i></div>`;

    card.innerHTML = `
      ${thumbHtml}
      <div class="alert-info">
        <div class="alert-header-row">
          <span class="alert-badge ${badgeClass}">${alert.threat_level}</span>
          <span class="alert-time">${alert.datetime_str ? alert.datetime_str.split(' ')[1] : ''}</span>
        </div>
        <div class="alert-headline">${escapeHtml(alert.alert_type)}</div>
        <div class="alert-desc">${escapeHtml(alert.message)}</div>
      </div>
    `;

    // Insert at top of feed
    alertFeed.insertBefore(card, alertFeed.firstChild);

    // Limit cards to 50
    while (alertFeed.children.length > 51) {
      alertFeed.removeChild(alertFeed.lastChild);
    }
  }

  clearAlertsBtn.addEventListener('click', () => {
    alertFeed.innerHTML = `
      <div class="no-alerts-placeholder" id="no-alerts-msg">
        <i class="fa-solid fa-shield-cat"></i>
        <div>Alert log cleared. Awaiting tactical incident events...</div>
      </div>
    `;
  });

  function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, (m) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[m]));
  }

  // --- Interactive Virtual Fence Drawing ---
  let isDrawing = false;
  let currentPoints = []; // [{x, y}] normalized 0.0 to 1.0
  const ctx = fenceCanvas.getContext('2d');

  function resizeCanvas() {
    const rect = videoWrapper.getBoundingClientRect();
    fenceCanvas.width = rect.width;
    fenceCanvas.height = rect.height;
    redrawCanvas();
  }

  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 500);

  toggleDrawBtn.addEventListener('click', () => {
    isDrawing = true;
    currentPoints = [];
    fenceCanvas.classList.add('drawing-active');
    drawInstruction.classList.remove('hidden');
    toggleDrawBtn.classList.add('hidden');
    cancelDrawBtn.classList.remove('hidden');
    saveDrawBtn.classList.remove('hidden');
    resizeCanvas();
  });

  cancelDrawBtn.addEventListener('click', exitDrawingMode);

  function exitDrawingMode() {
    isDrawing = false;
    currentPoints = [];
    fenceCanvas.classList.remove('drawing-active');
    drawInstruction.classList.add('hidden');
    toggleDrawBtn.classList.remove('hidden');
    cancelDrawBtn.classList.add('hidden');
    saveDrawBtn.classList.add('hidden');
    ctx.clearRect(0, 0, fenceCanvas.width, fenceCanvas.height);
  }

  fenceCanvas.addEventListener('click', (e) => {
    if (!isDrawing) return;
    const rect = fenceCanvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    // Store normalized coordinates [0.0, 1.0]
    const normX = Math.max(0, Math.min(1, px / fenceCanvas.width));
    const normY = Math.max(0, Math.min(1, py / fenceCanvas.height));
    currentPoints.push({ normX, normY, px, py });

    redrawCanvas();
  });

  function redrawCanvas() {
    ctx.clearRect(0, 0, fenceCanvas.width, fenceCanvas.height);
    if (!isDrawing || currentPoints.length === 0) return;

    ctx.strokeStyle = '#00f0ff';
    ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.lineWidth = 2;

    ctx.beginPath();
    currentPoints.forEach((pt, idx) => {
      const x = pt.normX * fenceCanvas.width;
      const y = pt.normY * fenceCanvas.height;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      // Draw vertex handle
      ctx.fillStyle = '#ff0055';
      ctx.fillRect(x - 4, y - 4, 8, 8);
      ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
    });

    if (currentPoints.length > 2) {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();
  }

  saveDrawBtn.addEventListener('click', () => {
    if (currentPoints.length < 3) {
      alert("Virtual fence requires at least 3 vertices to enclose a zone!");
      return;
    }
    fenceModal.classList.remove('hidden');
  });

  modalCloseBtn.addEventListener('click', () => fenceModal.classList.add('hidden'));
  modalCancelBtn.addEventListener('click', () => fenceModal.classList.add('hidden'));

  modalConfirmBtn.addEventListener('click', async () => {
    const name = fenceNameInput.value.trim() || `Tactical Zone ${Date.now() % 1000}`;
    const zoneType = fenceTypeSelect.value;
    const loiter = parseFloat(fenceLoiterInput.value) || 5.0;

    const polygon = currentPoints.map(pt => [roundNum(pt.normX, 4), roundNum(pt.normY, 4)]);
    const newFence = {
      id: `fence_${Date.now()}`,
      name: name,
      zone_type: zoneType,
      color: zoneType === 'RESTRICTED_PERIMETER' ? '#EF4444' : '#F59E0B',
      polygon: polygon,
      allowed_classes: zoneType === 'RESTRICTED_PERIMETER' ? [] : [0, 2, 7],
      loiter_threshold_sec: loiter
    };

    // Fetch existing fences and append
    try {
      const res = await fetch('/api/fences');
      const data = await res.json();
      const existing = data.fences || [];
      existing.push(newFence);

      await fetch('/api/fences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fences: existing })
      });

      fenceModal.classList.add('hidden');
      exitDrawingMode();
      loadFences();
    } catch (e) {
      alert("Error saving virtual fence: " + e);
    }
  });

  function roundNum(val, dec) {
    return Number(Math.round(val + 'e' + dec) + 'e-' + dec);
  }

  // --- Mode Toggle Handlers: Auto-Detect vs Manual Drawing ---
  if (modeManualBtn && modeAutoBtn) {
    modeManualBtn.addEventListener('click', () => switchPerimeterMode('manual'));
    modeAutoBtn.addEventListener('click', () => switchPerimeterMode('auto'));
  }
  if (rescanAutoBtn) {
    rescanAutoBtn.addEventListener('click', rescanAutoPerimeters);
  }

  async function switchPerimeterMode(mode) {
    if (mode === currentPerimeterMode) return;
    try {
      const res = await fetch('/api/fences/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
      });
      const data = await res.json();
      if (data.status === 'success') {
        currentPerimeterMode = mode;
        updateModeUI(mode);
        loadFences();
      }
    } catch (e) {
      console.error("Error toggling perimeter mode:", e);
    }
  }

  async function rescanAutoPerimeters() {
    try {
      rescanAutoBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> SCANNING...';
      const res = await fetch('/api/fences/auto_detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      await res.json();
      setTimeout(() => {
        rescanAutoBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> RE-SCAN SCENE';
        loadFences();
      }, 400);
    } catch (e) {
      rescanAutoBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> RE-SCAN SCENE';
      console.error("Auto detect scan error:", e);
    }
  }

  function updateModeUI(mode) {
    if (mode === 'auto') {
      if (modeAutoBtn) modeAutoBtn.classList.add('active');
      if (modeManualBtn) modeManualBtn.classList.remove('active');
      if (toggleDrawBtn) toggleDrawBtn.classList.add('hidden');
      if (rescanAutoBtn) rescanAutoBtn.classList.remove('hidden');
      if (isDrawing) exitDrawingMode();
    } else {
      if (modeManualBtn) modeManualBtn.classList.add('active');
      if (modeAutoBtn) modeAutoBtn.classList.remove('active');
      if (toggleDrawBtn) toggleDrawBtn.classList.remove('hidden');
      if (rescanAutoBtn) rescanAutoBtn.classList.add('hidden');
    }
  }

  // --- Load Active Virtual Fences ---
  async function loadFences() {
    try {
      const res = await fetch('/api/fences');
      const data = await res.json();
      const fences = data.fences || [];
      if (data.mode) {
        currentPerimeterMode = data.mode;
        updateModeUI(data.mode);
      }

      fenceTagsContainer.innerHTML = '';
      fences.forEach(f => {
        const tag = document.createElement('span');
        tag.className = 'fence-tag';
        tag.style.borderColor = f.color || '#EF4444';
        const autoBadge = f.is_auto ? '<span style="font-size:9px;background:#00f0ff;color:#000;padding:1px 4px;border-radius:2px;margin-right:6px;font-weight:bold;">AUTO</span>' : '';
        tag.innerHTML = `${autoBadge}<i class="fa-solid fa-shield"></i> ${escapeHtml(f.name)}`;
        fenceTagsContainer.appendChild(tag);
      });
    } catch (e) {
      console.warn("Fences fetch:", e);
    }
  }

  // --- Telemetry Poller ---
  async function pollStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();

      // GPU & VRAM
      const gpu = data.gpu;
      gpuNameText.textContent = gpu.name || 'NVIDIA RTX 3050';
      vramValText.textContent = `${gpu.vram_used_mb} / ${gpu.vram_total_mb} MB (${gpu.vram_percent}%)`;
      vramProgress.style.width = `${Math.min(100, gpu.vram_percent)}%`;

      // CPU & RAM
      const sys = data.system;
      cpuPercentText.textContent = `${sys.cpu_percent}%`;
      cpuProgress.style.width = `${Math.min(100, sys.cpu_percent)}%`;
      ramValText.textContent = `${sys.ram_used_gb} / ${sys.ram_total_gb} GB (${sys.ram_percent}%)`;

      // Pipeline
      const pipe = data.pipeline;
      streamFps.textContent = pipe.fps.toFixed(1);
      streamTracks.textContent = pipe.active_tracks;

      // Zero-DCE Night Vision Indicator
      if (pipe.zero_dce_active) {
        zerodceBadge.style.background = 'rgba(0, 255, 128, 0.2)';
        zerodceBadge.style.borderColor = '#00ff80';
        zerodceBadge.style.color = '#00ff80';
        zerodceStatus.textContent = 'ZERO-DCE: ACTIVE (BOOSTING)';
      } else {
        zerodceBadge.style.background = 'rgba(0, 240, 255, 0.12)';
        zerodceBadge.style.borderColor = '#00f0ff';
        zerodceBadge.style.color = '#00f0ff';
        zerodceStatus.textContent = 'ZERO-DCE: STANDBY';
      }
    } catch (e) {
      // Offline / reconnecting
    }
  }

  // Initial loads & polling loops
  loadFences();
  setInterval(pollStats, 2000);
  pollStats();
});
