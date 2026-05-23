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

  let html = `
    <div class="result-content">
      <div class="result-summary">
        <h3>Detection Results</h3>
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
