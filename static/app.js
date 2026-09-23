document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('xray-canvas');
  const ctx = canvas.getContext('2d');
  
  const samplePills = document.querySelectorAll('.sample-pill');
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const btnPredict = document.getElementById('btn-predict');

  const previewFilename = document.getElementById('preview-filename');
  const previewPatientId = document.getElementById('preview-patient-id');
  const resultCard = document.getElementById('result-card');

  let currentImageData = null;
  let currentSamplePath = null;
  let currentPatientId = 'person19';

  // Draw default synthetic X-ray on canvas
  function renderCanvasXray(isPneumonia = false) {
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = '#0a0e17';
    ctx.fillRect(0, 0, w, h);

    // Spine
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.fillRect(w * 0.46, h * 0.1, w * 0.08, h * 0.8);

    // Ribs
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

    // Lungs
    ctx.fillStyle = 'rgba(10, 15, 25, 0.9)';
    ctx.beginPath();
    ctx.ellipse(w * 0.32, h * 0.48, 30, 45, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(w * 0.68, h * 0.48, 30, 45, 0, 0, Math.PI * 2);
    ctx.fill();

    // Pneumonia Infiltrate patch
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

  renderCanvasXray(true); // default pneumonia sample

  // Fetch CV summary metrics from backend API
  fetch('/api/cv_summary')
    .then(res => res.json())
    .then(data => {
      if (data && data.summary) {
        document.getElementById('val-accuracy').textContent = `${(data.summary.mean_accuracy * 100).toFixed(1)}%`;
        document.getElementById('val-sensitivity').textContent = `${(data.summary.mean_sensitivity * 100).toFixed(1)}%`;
        document.getElementById('val-specificity').textContent = `${(data.summary.mean_specificity * 100).toFixed(1)}%`;
        
        if (data.optimal_threshold) {
          document.getElementById('val-threshold').textContent = data.optimal_threshold.optimal_threshold.toFixed(2);
        }

        // Render Fold Table
        if (data.folds) {
          const tbody = document.getElementById('cv-table-body');
          tbody.innerHTML = '';
          data.folds.forEach(f => {
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
    .catch(err => console.log('Using default metric view'));

  // Sample Selection
  samplePills.forEach(pill => {
    pill.addEventListener('click', () => {
      samplePills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');

      const sampleType = pill.dataset.sample;
      currentImageData = null;

      if (sampleType === 'normal-1') {
        previewFilename.textContent = 'person100_normal_1.jpeg';
        currentPatientId = 'person100';
        previewPatientId.innerHTML = `<i class="fa-solid fa-id-card"></i> Patient Subject: <strong>${currentPatientId}</strong>`;
        renderCanvasXray(false);
      } else if (sampleType === 'pneumonia-1') {
        previewFilename.textContent = 'person19_bacteria_62.jpeg';
        currentPatientId = 'person19';
        previewPatientId.innerHTML = `<i class="fa-solid fa-id-card"></i> Patient Subject: <strong>${currentPatientId}</strong>`;
        renderCanvasXray(true);
      } else {
        previewFilename.textContent = 'person44_virus_112.jpeg';
        currentPatientId = 'person44';
        previewPatientId.innerHTML = `<i class="fa-solid fa-id-card"></i> Patient Subject: <strong>${currentPatientId}</strong>`;
        renderCanvasXray(true);
      }
    });
  });

  // File Input Handler
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      previewFilename.textContent = file.name;
      currentPatientId = 'person_uploaded';
      previewPatientId.innerHTML = `<i class="fa-solid fa-id-card"></i> Patient Subject: <strong>Upload User</strong>`;

      const reader = new FileReader();
      reader.onload = (evt) => {
        currentImageData = evt.target.result;
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = evt.target.result;
      };
      reader.readAsDataURL(file);
    }
  });

  // Prediction Button Handler
  btnPredict.addEventListener('click', () => {
    btnPredict.disabled = true;
    btnPredict.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running 5-Fold Ensemble...`;

    const payload = {
      image_data: currentImageData,
      sample_path: currentSamplePath,
      patient_id: currentPatientId
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

    // Render 5-Fold Bars
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
});
