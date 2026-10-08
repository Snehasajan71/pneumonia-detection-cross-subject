document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('medical-canvas');
  const ctx = canvas.getContext('2d');
  
  const modalityBtns = document.querySelectorAll('.modality-btn');
  const sampleGallery = document.getElementById('sample-gallery');
  const fileInput = document.getElementById('file-input');
  const btnPredict = document.getElementById('btn-predict');

  const workspaceView = document.getElementById('workspace-view');
  const comparisonView = document.getElementById('comparison-view');
  const metricsBanner = document.getElementById('metrics-banner');

  const workspaceTitle = document.getElementById('workspace-title');
  const modalityTag = document.getElementById('modality-tag');
  const uploadModalityLabel = document.getElementById('upload-modality-label');
  const previewFilename = document.getElementById('preview-filename');
  const previewPatientId = document.getElementById('preview-patient-id');
  const resultCard = document.getElementById('result-card');
  const subjectPrefixCode = document.getElementById('subject-prefix-code');

  let currentModality = 'xray'; // 'xray' | 'ct_scan' | 'comparison'
  let currentImageData = null;
  let currentSamplePath = null;
  let currentPatientId = 'person19';

  // Canvas renderers
  function renderCanvas(modality = 'xray', isPneumonia = true) {
    const w = canvas.width;
    const h = canvas.height;

    if (modality === 'ct_scan') {
      // Axial CT Scan Cross-Section Renderer
      ctx.fillStyle = '#05070c';
      ctx.fillRect(0, 0, w, h);

      // Thoracic body ring (soft tissue & rib border)
      ctx.fillStyle = '#2a354a';
      ctx.beginPath();
      ctx.ellipse(w/2, h/2, w*0.42, h*0.40, 0, 0, Math.PI*2);
      ctx.fill();

      // Lung Parenchyma (dark air spaces)
      ctx.fillStyle = '#090d16';
      ctx.beginPath();
      ctx.ellipse(w*0.35, h/2, w*0.14, h*0.26, 0, 0, Math.PI*2);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(w*0.65, h/2, w*0.14, h*0.26, 0, 0, Math.PI*2);
      ctx.fill();

      // Spine & Mediastinum (center bright tissue)
      ctx.fillStyle = '#4f5e7b';
      ctx.beginPath();
      ctx.arc(w/2, h*0.22, 14, 0, Math.PI*2);
      ctx.fill();
      ctx.fillRect(w*0.46, h*0.70, w*0.08, h*0.18);

      // Ground-Glass Opacity (GGO) / Consolidation patch in CT
      if (isPneumonia) {
        const grad = ctx.createRadialGradient(w*0.34, h*0.55, 3, w*0.34, h*0.55, 24);
        grad.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
        grad.addColorStop(0.6, 'rgba(220, 220, 240, 0.45)');
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(w*0.34, h*0.55, 24, 0, Math.PI*2);
        ctx.fill();
      }
    } else {
      // Chest X-Ray Projection Renderer
      ctx.fillStyle = '#0a0e17';
      ctx.fillRect(0, 0, w, h);

      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.fillRect(w * 0.46, h * 0.1, w * 0.08, h * 0.8);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 3;
      for (let y = 30; y < h - 30; y += 22) {
        ctx.beginPath();
        ctx.arc(w * 0.3, y, 35, -0.4, 0.8);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(w * 0.7, y, 35, 2.3, 3.5);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(10, 15, 25, 0.9)';
      ctx.beginPath();
      ctx.ellipse(w * 0.32, h * 0.48, 30, 45, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(w * 0.68, h * 0.48, 30, 45, 0, 0, Math.PI * 2);
      ctx.fill();

      if (isPneumonia) {
        const grad = ctx.createRadialGradient(w * 0.35, h * 0.55, 5, w * 0.35, h * 0.55, 28);
        grad.addColorStop(0, 'rgba(255, 255, 255, 0.85)');
        grad.addColorStop(0.5, 'rgba(255, 255, 255, 0.4)');
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(w * 0.35, h * 0.55, 28, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Load CV metrics for current modality
  function loadModalityMetrics(modality) {
    fetch(`/api/cv_summary?modality=${modality}`)
      .then(res => res.json())
      .then(data => {
        if (data) {
          document.getElementById('val-accuracy').textContent = `${(data.mean_accuracy * 100).toFixed(1)}%`;
          document.getElementById('val-sensitivity').textContent = `${(data.mean_sensitivity * 100).toFixed(1)}%`;
          document.getElementById('val-specificity').textContent = `${(data.mean_specificity * 100).toFixed(1)}%`;
          
          if (data.optimal_threshold) {
            document.getElementById('val-threshold').textContent = data.optimal_threshold.optimal_threshold.toFixed(2);
          }

          if (data.fold_results) {
            const tbody = document.getElementById('cv-table-body');
            tbody.innerHTML = '';
            data.fold_results.forEach(f => {
              const tr = document.createElement('tr');
              tr.innerHTML = `
                <td>Fold ${f.fold}</td>
                <td>${f.num_val_subjects} Patients</td>
                <td><strong>${(f.accuracy * 100).toFixed(1)}%</strong></td>
                <td>${f.f1.toFixed(3)}</td>
                <td>${(f.recall * 100).toFixed(1)}%</td>
                <td>${(f.specificity * 100).toFixed(1)}%</td>
              `;
              tbody.appendChild(tr);
            });
          }
        }
      })
      .catch(err => console.log('Metric load fallback'));
  }

  // Load sample gallery for modality
  function loadSamples(modality) {
    fetch(`/api/samples?modality=${modality}`)
      .then(res => res.json())
      .then(samples => {
        sampleGallery.innerHTML = '';
        samples.forEach((s, idx) => {
          const pill = document.createElement('div');
          pill.className = `sample-pill ${idx === 0 ? 'active' : ''}`;
          const isPneu = s.category === 'PNEUMONIA';
          pill.innerHTML = `
            <i class="fa-solid ${isPneu ? 'fa-triangle-exclamation text-danger' : 'fa-circle-check text-success'}"></i>
            ${s.filename}
          `;
          pill.addEventListener('click', () => {
            document.querySelectorAll('.sample-pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentImageData = null;
            currentSamplePath = s.filepath;
            currentPatientId = s.subject_id;
            previewFilename.textContent = s.filename;
            previewPatientId.innerHTML = `<i class="fa-solid fa-id-card"></i> Patient Subject: <strong>${s.subject_id}</strong>`;
            renderCanvas(modality, isPneu);
          });
          sampleGallery.appendChild(pill);
        });

        if (samples.length > 0) {
          const first = samples[0];
          currentSamplePath = first.filepath;
          currentPatientId = first.subject_id;
          previewFilename.textContent = first.filename;
          previewPatientId.innerHTML = `<i class="fa-solid fa-id-card"></i> Patient Subject: <strong>${first.subject_id}</strong>`;
          renderCanvas(modality, first.category === 'PNEUMONIA');
        }
      });
  }

  // Load Comparison Data
  function loadComparisonView() {
    fetch('/api/comparison')
      .then(res => res.json())
      .then(data => {
        if (data && data.metrics_comparison) {
          const tbody = document.getElementById('comparison-table-body');
          tbody.innerHTML = '';
          data.metrics_comparison.forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><strong>${m.metric}</strong></td>
              <td>${m.xray}</td>
              <td>${m.ct_scan}</td>
              <td><span class="winner-tag"><i class="fa-solid fa-trophy"></i> ${m.winner}</span></td>
            `;
            tbody.appendChild(tr);
          });
        }
      });
  }

  // Tab Switch Handler
  modalityBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modalityBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const selected = btn.dataset.modality;
      currentModality = selected;

      if (selected === 'comparison') {
        workspaceView.classList.add('hidden');
        comparisonView.classList.remove('hidden');
        metricsBanner.classList.add('hidden');
        loadComparisonView();
      } else {
        workspaceView.classList.remove('hidden');
        comparisonView.classList.add('hidden');
        metricsBanner.classList.remove('hidden');

        workspaceTitle.textContent = selected === 'ct_scan' ? 'Chest CT Scan Prediction' : 'Chest X-Ray Prediction';
        modalityTag.textContent = selected === 'ct_scan' ? 'CT SCAN ENSEMBLE' : 'X-RAY ENSEMBLE';
        uploadModalityLabel.textContent = selected === 'ct_scan' ? 'Chest CT Scan' : 'Chest X-Ray';
        subjectPrefixCode.textContent = selected === 'ct_scan' ? 'ct_patient{ID}' : 'person{ID}';

        loadModalityMetrics(selected);
        loadSamples(selected);
      }
    });
  });

  // File Upload Handler
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      previewFilename.textContent = file.name;
      currentPatientId = 'uploaded_patient';
      previewPatientId.innerHTML = `<i class="fa-solid fa-id-card"></i> Patient Subject: <strong>Upload User</strong>`;

      const reader = new FileReader();
      reader.onload = (evt) => {
        currentImageData = evt.target.result;
        currentSamplePath = null;
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = evt.target.result;
      };
      reader.readAsDataURL(file);
    }
  });

  // Predict Handler
  btnPredict.addEventListener('click', () => {
    btnPredict.disabled = true;
    btnPredict.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running 5-Fold ${currentModality.toUpperCase()} Ensemble...`;

    const payload = {
      image_data: currentImageData,
      sample_path: currentSamplePath,
      modality: currentModality
    };

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Run Best Prediction Ensemble`;
      renderPredictionResult(data);
    })
    .catch(err => {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Run Best Prediction Ensemble`;
      alert('Prediction Error: ' + err);
    });
  });

  function renderPredictionResult(data) {
    resultCard.classList.remove('hidden');
    const clsElem = document.getElementById('result-class');
    const badgeElem = document.getElementById('result-risk-badge');
    const confText = document.getElementById('result-confidence-text');
    const fillElem = document.getElementById('result-progress-bar');
    const foldBarsContainer = document.getElementById('fold-bars-container');

    clsElem.textContent = data.predicted_class;
    confText.textContent = `${data.confidence_percent}%`;
    fillElem.style.width = `${data.confidence_percent}%`;

    if (data.predicted_class === 'PNEUMONIA') {
      clsElem.className = 'result-class pneumonia';
      badgeElem.className = 'risk-badge';
      badgeElem.textContent = data.risk_category || 'HIGH RISK';
    } else {
      clsElem.className = 'result-class normal';
      badgeElem.className = 'risk-badge normal';
      badgeElem.textContent = 'NORMAL - LOW RISK';
    }

    foldBarsContainer.innerHTML = '';
    if (data.fold_probabilities) {
      data.fold_probabilities.forEach((p, idx) => {
        const item = document.createElement('div');
        item.className = 'fold-bar-item';
        const heightPct = Math.max(p * 100, 10);
        item.innerHTML = `
          <span>Fold ${idx + 1}</span>
          <div class="fold-bar-fill" style="height: ${heightPct}%;"></div>
          <strong>${(p * 100).toFixed(0)}%</strong>
        `;
        foldBarsContainer.appendChild(item);
      });
    }
  }

  // Initial Load
  loadModalityMetrics('xray');
  loadSamples('xray');
});
