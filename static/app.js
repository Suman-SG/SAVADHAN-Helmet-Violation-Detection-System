function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function showProgress(visible) {
  const progressBar = document.getElementById('progress-bar');
  if (!progressBar) return;
  
  if (visible) {
    progressBar.classList.add('active');
    progressBar.style.width = '30%';
    setTimeout(() => {
      progressBar.style.width = '60%';
    }, 100);
  } else {
    progressBar.style.width = '100%';
    setTimeout(() => {
      progressBar.classList.remove('active');
      progressBar.style.width = '0%';
    }, 300);
  }
}

// Image Preview for images
document.addEventListener('DOMContentLoaded', () => {
  // mobile nav toggle
  const navToggle = document.getElementById('nav-toggle');
  const topnav = document.getElementById('topnav');
  if (navToggle && topnav) {
    navToggle.addEventListener('click', (e) => {
      e.preventDefault();
      const open = topnav.classList.toggle('show');
      navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    // close on outside click
    document.addEventListener('click', (ev) => {
      if (!topnav.contains(ev.target) && !navToggle.contains(ev.target)) topnav.classList.remove('show');
    });
  }
  const imageForm = document.getElementById('image-form');
  const imageDrop = document.getElementById('image-drop');
  const imageInput = imageDrop?.querySelector('input[type="file"]');
  const imagePreviewContainer = document.getElementById('image-preview-container');
  const imagePreview = document.getElementById('image-preview');
  const previewName = document.getElementById('preview-name');

  if (imageInput) {
    imageInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (event) => {
          imagePreview.src = event.target.result;
          previewName.textContent = file.name;
          imagePreviewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
      }
    });

    // Drag and drop
    imageDrop.addEventListener('dragover', (e) => {
      e.preventDefault();
      imageDrop.style.background = 'rgba(77, 214, 184, 0.1)';
    });

    imageDrop.addEventListener('dragleave', () => {
      imageDrop.style.background = '';
    });

    imageDrop.addEventListener('drop', (e) => {
      e.preventDefault();
      imageDrop.style.background = '';
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        imageInput.files = files;
        imageInput.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
  }

  // Video Preview
  const videoForm = document.getElementById('video-form');
  const videoDrop = document.getElementById('video-drop');
  const videoInput = videoDrop?.querySelector('input[type="file"]');
  const videoPreviewContainer = document.getElementById('video-preview-container');
  const videoPreviewName = document.getElementById('video-preview-name');

  if (videoInput) {
    videoInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        videoPreviewName.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        videoPreviewContainer.style.display = 'block';
      }
    });

    // Drag and drop
    videoDrop.addEventListener('dragover', (e) => {
      e.preventDefault();
      videoDrop.style.background = 'rgba(77, 214, 184, 0.1)';
    });

    videoDrop.addEventListener('dragleave', () => {
      videoDrop.style.background = '';
    });

    videoDrop.addEventListener('drop', (e) => {
      e.preventDefault();
      videoDrop.style.background = '';
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        videoInput.files = files;
        videoInput.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
  }

  // Form submission handlers
  if (imageForm) {
    imageForm.addEventListener('submit', handleFormSubmit);
  }

  if (videoForm) {
    videoForm.addEventListener('submit', handleFormSubmit);
  }
});

async function handleFormSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const endpoint = form.getAttribute('data-endpoint');
  const isVideo = form.id === 'video-form';
  const resultPanel = document.getElementById(isVideo ? 'video-result' : 'image-result');
  const submitBtn = form.querySelector('button[type="submit"]');

  showProgress(true);
  submitBtn.disabled = true;
  submitBtn.textContent = isVideo ? '🎥 Processing...' : '📷 Processing...';

  const formData = new FormData(form);

  try {
    const response = await fetch(endpoint, { method: 'POST', body: formData });
    const data = await response.json();

    if (isVideo) {
      renderVideoResult(data, resultPanel);
    } else {
      renderImageResult(data, resultPanel);
    }
  } catch (error) {
    resultPanel.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
  } finally {
    showProgress(false);
    submitBtn.disabled = false;
    submitBtn.textContent = isVideo ? '🎥 Run Video Detection' : '📷 Run Image Detection';
  }
}

function renderImageResult(data, panel) {
  if (!data.success) {
    panel.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.message)}</div>`;
    return;
  }

  const totalRiders = Number(data.summary?.total_riders || 0);
  const violations = Number(data.summary?.violations || 0);
  const avgConf = Number(data.summary?.avg_conf || 0);
  const weakDetection = totalRiders > 0 && totalRiders <= 1 && avgConf > 0 && avgConf < 55 && violations === 0;
  const noRider = totalRiders === 0;
  const resultTone = noRider || weakDetection ? 'warning' : (violations > 0 ? 'danger' : 'success');
  const resultHeadline = noRider
    ? 'No clear rider detected'
    : weakDetection
      ? 'Possible false positive'
      : violations > 0
        ? 'Violation detected'
        : 'Safe rider detected';
  const resultCopy = noRider
    ? 'This image does not look like a valid rider/helmet scene. Please upload a clear bike or scooter image.'
    : weakDetection
      ? `The model found a weak match (${avgConf.toFixed(1)}%). This may be a wrong image, so please retry with a clearer bike/rider photo.`
      : 'The image was processed successfully.';

  let html = `
    <div class="result-content">
      <div class="result-summary result-summary-${resultTone}">
        <div class="section-label">${escapeHtml(resultHeadline)}</div>
        <h3>${escapeHtml(resultCopy)}</h3>
        <div class="summary-stats">
          <div class="summary-stat">
            <span class="stat-icon">👥</span>
            <div><span class="stat-label">Riders</span><span class="stat-value">${data.summary.total_riders}</span></div>
          </div>
          <div class="summary-stat">
            <span class="stat-icon">✅</span>
            <div><span class="stat-label">Safe</span><span class="stat-value">${data.summary.safe_riders}</span></div>
          </div>
          <div class="summary-stat">
            <span class="stat-icon">🚨</span>
            <div><span class="stat-label">Violations</span><span class="stat-value">${data.summary.violations}</span></div>
          </div>
          <div class="summary-stat">
            <span class="stat-icon">🎯</span>
            <div><span class="stat-label">Confidence</span><span class="stat-value">${avgConf ? avgConf.toFixed(1) + '%' : '—'}</span></div>
          </div>
        </div>
      </div>
  `;

  if (data.annotated_url) {
    html += `
      <div class="result-image-section">
        <h4>Annotated Image</h4>
        <img src="${escapeHtml(data.annotated_url)}" alt="Annotated result" class="result-image">
      </div>
    `;
  }

  if (data.records && data.records.length > 0) {
    html += '<div class="result-records"><h4>Detected Plates & Violations</h4>';
    data.records.forEach(record => {
      const plateText = escapeHtml(record.plate_text || 'NOT_DETECTED');
      const emailStatus = record.email_sent ? '📧 Sent' : '❌ Not Sent';
      const violationId = escapeHtml(record.violation_id || '—');
      const ownerName = escapeHtml(record.owner_name || 'Unknown');
      
      html += `
        <div class="record-card">
          <div class="record-header">
            <span class="plate-badge">${plateText}</span>
            <span class="email-status ${record.email_sent ? 'sent' : 'unsent'}">${emailStatus}</span>
          </div>
          <div class="record-details">
            <p><strong>Owner:</strong> ${ownerName}</p>
            <p><strong>Violations:</strong> ${record.violation_count}</p>
            <p><strong>Confidence:</strong> ${(record.ocr_conf * 100).toFixed(1)}%</p>
            <p><strong>Status:</strong> ${escapeHtml(record.status || 'Pending')}</p>
          </div>
      `;
      
      if (record.evidence_url) {
        html += `<a href="${escapeHtml(record.evidence_url)}" target="_blank" class="btn btn-small">📸 View Evidence</a>`;
      }
      
      html += '</div>';
    });
    html += '</div>';
  }

  html += '</div>';
  panel.innerHTML = html;
}

function renderVideoResult(data, panel) {
  if (!data.success) {
    panel.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.message)}</div>`;
    return;
  }

  let html = `
    <div class="result-content">
      <div class="result-summary">
        <h3>Video Processing Complete</h3>
        <div class="summary-stats">
          <div class="summary-stat">
            <span class="stat-icon">📹</span>
            <div><span class="stat-label">Frames Processed</span><span class="stat-value">${data.processed_frames}</span></div>
          </div>
          <div class="summary-stat">
            <span class="stat-icon">🚨</span>
            <div><span class="stat-label">Violations</span><span class="stat-value">${data.violations}</span></div>
          </div>
          <div class="summary-stat">
            <span class="stat-icon">📋</span>
            <div><span class="stat-label">Unique Plates</span><span class="stat-value">${data.plates.length}</span></div>
          </div>
        </div>
      </div>
  `;

  if (data.plates.length > 0) {
    html += '<div class="plates-section"><h4>Detected Plates</h4><div class="plate-list">';
    data.plates.forEach(plate => {
      html += `<span class="plate-badge">${escapeHtml(plate)}</span>`;
    });
    html += '</div></div>';
  }

  if (data.frames && data.frames.length > 0) {
    html += '<div class="frames-section"><h4>Frame Details</h4>';
    data.frames.forEach((frame, idx) => {
      html += `
        <div class="frame-card">
          <div class="frame-info">
            <span class="frame-name">${escapeHtml(frame.frame_name)}</span>
            <span class="frame-violations">🚨 ${frame.violations}</span>
            <span class="frame-riders">👥 ${frame.riders}</span>
            ${frame.email_sent ? '<span class="email-sent">📧</span>' : ''}
          </div>
      `;
      if (frame.annotated_url) {
        html += `<img src="${escapeHtml(frame.annotated_url)}" alt="Frame ${idx}" class="frame-thumbnail">`;
      }
      html += '</div>';
    });
    html += '</div>';
  }

  html += '</div>';
  panel.innerHTML = html;
}

// --- Camera capture & recording support ---
let _mediaStream = null;
let _mediaRecorder = null;
let _recordedChunks = [];

async function openCamera(photoMode = true) {
  const container = document.getElementById('camera-container');
  const streamEl = document.getElementById('camera-stream');
  const controls = document.getElementById('camera-controls');
  const closeBtn = document.getElementById('close-camera');

  try {
    _mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: !photoMode });
    streamEl.srcObject = _mediaStream;
    container.style.display = 'block';
    streamEl.style.display = 'block';
    controls.style.display = 'flex';
    closeBtn.style.display = 'inline-block';
    document.getElementById('camera-controls').dataset.mode = photoMode ? 'photo' : 'video';
  } catch (err) {
    alert('Unable to access camera: ' + err.message);
  }
}

function closeCamera() {
  if (_mediaStream) {
    _mediaStream.getTracks().forEach(t => t.stop());
    _mediaStream = null;
  }
  const container = document.getElementById('camera-container');
  const streamEl = document.getElementById('camera-stream');
  const controls = document.getElementById('camera-controls');
  const closeBtn = document.getElementById('close-camera');
  streamEl.srcObject = null;
  container.style.display = 'none';
  controls.style.display = 'none';
  closeBtn.style.display = 'none';
}

function capturePhotoAndUpload() {
  const video = document.getElementById('camera-stream');
  const canvas = document.getElementById('camera-canvas');
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async (blob) => {
    // show preview
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    imagePreview.src = URL.createObjectURL(blob);
    document.getElementById('preview-name').textContent = 'Camera capture';
    imagePreviewContainer.style.display = 'block';

    // upload to same image endpoint
    const endpoint = document.getElementById('image-form').getAttribute('data-endpoint');
    const fd = new FormData();
    fd.append('media', blob, 'capture.jpg');
    showProgress(true);
    try {
      const resp = await fetch(endpoint, { method: 'POST', body: fd });
      const data = await resp.json();
      renderImageResult(data, document.getElementById('image-result'));
    } catch (err) {
      document.getElementById('image-result').innerHTML = `<div class="error-message">${escapeHtml(err.message)}</div>`;
    } finally {
      showProgress(false);
    }
  }, 'image/jpeg', 0.9);
}

function startRecording() {
  if (!_mediaStream) return alert('Camera not started');
  _recordedChunks = [];
  _mediaRecorder = new MediaRecorder(_mediaStream, { mimeType: 'video/webm;codecs=vp8,opus' });
  _mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size) _recordedChunks.push(e.data); };
  _mediaRecorder.onstop = uploadRecordedVideo;
  _mediaRecorder.start();
  document.getElementById('start-record').style.display = 'none';
  document.getElementById('stop-record').style.display = 'inline-block';
}

function stopRecording() {
  if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
    _mediaRecorder.stop();
  }
}

async function uploadRecordedVideo() {
  const blob = new Blob(_recordedChunks, { type: 'video/webm' });
  const endpoint = document.getElementById('video-form').getAttribute('data-endpoint');
  const fd = new FormData();
  fd.append('media', blob, 'recording.webm');
  // include default video params
  fd.append('frame_stride', document.querySelector('#video-form input[name="frame_stride"]').value || 20);
  fd.append('max_frames', document.querySelector('#video-form input[name="max_frames"]').value || 40);

  showProgress(true);
  try {
    const resp = await fetch(endpoint, { method: 'POST', body: fd });
    const data = await resp.json();
    renderVideoResult(data, document.getElementById('video-result'));
  } catch (err) {
    document.getElementById('video-result').innerHTML = `<div class="error-message">${escapeHtml(err.message)}</div>`;
  } finally {
    showProgress(false);
    document.getElementById('start-record').style.display = 'inline-block';
    document.getElementById('stop-record').style.display = 'none';
    closeCamera();
  }
}

// connect camera controls after DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const openPhoto = document.getElementById('open-camera-photo');
  const openVideo = document.getElementById('open-camera-video');
  const closeBtn = document.getElementById('close-camera');
  const captureBtn = document.getElementById('capture-photo');
  const startBtn = document.getElementById('start-record');
  const stopBtn = document.getElementById('stop-record');

  if (openPhoto) openPhoto.addEventListener('click', (e) => { e.preventDefault(); openCamera(true); });
  if (openVideo) openVideo.addEventListener('click', (e) => { e.preventDefault(); openCamera(false); });
  if (closeBtn) closeBtn.addEventListener('click', (e) => { e.preventDefault(); closeCamera(); });
  if (captureBtn) captureBtn.addEventListener('click', (e) => { e.preventDefault(); capturePhotoAndUpload(); });
  if (startBtn) startBtn.addEventListener('click', (e) => { e.preventDefault(); startRecording(); });
  if (stopBtn) stopBtn.addEventListener('click', (e) => { e.preventDefault(); stopRecording(); });
});

// Admin verification actions (confirm / reject)
document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('confirm-modal');
  const modalBody = document.getElementById('confirm-modal-body');
  const modalTitle = document.getElementById('confirm-modal-title');
  const modalCancel = document.getElementById('confirm-modal-cancel');
  const modalConfirm = document.getElementById('confirm-modal-confirm');
  const toast = document.getElementById('toast');

  let pendingAction = null;

  function showModal(title, body) {
    modalTitle.textContent = title;
    modalBody.textContent = body;
    modal.setAttribute('aria-hidden', 'false');
  }

  function hideModal() { modal.setAttribute('aria-hidden', 'true'); pendingAction = null; }

  function showToast(msg, timeout = 2400) {
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), timeout);
  }

  modalCancel.addEventListener('click', (e) => { e.preventDefault(); hideModal(); });

  modalConfirm.addEventListener('click', async (e) => {
    e.preventDefault();
    if (!pendingAction) return hideModal();
    const { id, action, row } = pendingAction;
    modalConfirm.disabled = true;
    modalConfirm.textContent = action === 'confirm' ? 'Confirming...' : 'Rejecting...';

    try {
      const fd = new FormData();
      fd.append('violation_id', id);
      fd.append('action', action);
      const resp = await fetch('/admin/violation/verify', { method: 'POST', body: fd });
      const data = await resp.json();
      if (resp.ok && data.success) {
        // Update the row inline: email column (index 5), invoice (index 6)
        const cells = row.querySelectorAll('td');
        if (cells.length >= 7) {
          cells[5].textContent = data.email_sent ? 'Yes' : 'No';
          cells[6].textContent = data.invoice_number || cells[6].textContent || '';
        }
        // visually confirm
        row.style.transition = 'background 0.4s ease';
        row.style.background = action === 'confirm' ? 'linear-gradient(90deg, rgba(82,211,155,0.06), transparent)' : 'linear-gradient(90deg, rgba(255,111,121,0.04), transparent)';
        setTimeout(() => row.style.background = '', 2200);
        showToast(data.message || 'Updated');
      } else {
        showToast(data.message || 'Failed to update');
      }
    } catch (err) {
      showToast('Request failed: ' + err.message);
    } finally {
      modalConfirm.disabled = false;
      modalConfirm.textContent = 'Confirm';
      hideModal();
    }
  });

  document.querySelectorAll('.verify-violation').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const id = btn.dataset.id;
      const action = btn.dataset.action;
      if (!id || !action) return;
      const row = btn.closest('tr');
      pendingAction = { id, action, row };
      showModal(`Mark ${id} as ${action.toUpperCase()}?`, `This will ${action === 'confirm' ? 'confirm and attempt to send the email/invoice' : 'mark as rejected — no email will be sent'}.`);
    });
  });
});
