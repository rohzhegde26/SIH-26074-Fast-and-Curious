/**
 * frontend/app.js
 * Mandya Weather Advisory PWA — Offline-First Client Engine
 */

const DB_NAME = "mandya-weather-db";
const STORE = "forecasts";

// IndexedDB Helper
function getDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 2);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "lgd_code" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveAllToDb(records) {
  if (!records || !records.length) return;
  const db = await getDb();
  const tx = db.transaction(STORE, "readwrite");
  const store = tx.objectStore(STORE);
  records.forEach(r => store.put(r));
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function getAllFromDb() {
  const db = await getDb();
  const tx = db.transaction(STORE, "readonly");
  const request = tx.objectStore(STORE).getAll();
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => reject(request.error);
  });
}

// Application State
let currentRecords = [];
let selectedLgdCode = null;
let currentLanguage = "en";
let mapPaths = new Map();
let currentZoom = 1.0;
let panOffset = { x: 0, y: 0 };
let initialViewBox = null;

// TopoJSON Arc Decoder
function decodeArc(topology, index) {
  const points = [];
  let x = 0;
  let y = 0;
  const arc = topology.arcs[index < 0 ? ~index : index];
  for (const pair of arc) {
    x += pair[0];
    y += pair[1];
    points.push([
      x * topology.transform.scale[0] + topology.transform.translate[0],
      y * topology.transform.scale[1] + topology.transform.translate[1]
    ]);
  }
  return index < 0 ? points.reverse() : points;
}

function pathStringFor(topology, rings) {
  return rings.map(ring => {
    const points = ring.flatMap((arcIdx, i) => decodeArc(topology, arcIdx).slice(i ? 1 : 0));
    return points.length ? `M${points.map(p => `${p[0]},${-p[1]}`).join("L")}Z` : "";
  }).join("");
}

// Color Choropleth Palette
function getRainColor(mm) {
  if (mm > 64.4) return "#0d47a1"; // Heavy
  if (mm > 15.5) return "#1e88e5"; // Moderate
  if (mm > 2.5)  return "#90caf9"; // Light
  return "#e3f2fd";                // Dry (<2.5mm)
}

function getIntensityLabel(mm) {
  if (mm > 64.4) return { label: "Heavy Rain", class: "pill-heavy" };
  if (mm > 15.5) return { label: "Moderate Rain", class: "pill-moderate" };
  if (mm > 2.5)  return { label: "Light Rain", class: "pill-light" };
  return { label: "Dry / Trace", class: "pill-dry" };
}

// Render Localized Forecast & Advisories
function renderForecastDetails(record) {
  if (!record) return;
  selectedLgdCode = record.lgd_code;

  // Highlight selected map polygon
  mapPaths.forEach((path, code) => {
    path.classList.toggle("selected", code === record.lgd_code);
  });

  const container = document.querySelector("#forecast-details");
  const exp = record.rainfall_mm.expected;
  const lMin = record.rainfall_mm.likely_min;
  const lMax = record.rainfall_mm.likely_max;
  const intensity = getIntensityLabel(exp);

  const ragiAdv = currentLanguage === "kn" ? record.advisory.ragi.action_kn : record.advisory.ragi.action_en;
  const paddyAdv = currentLanguage === "kn" ? record.advisory.paddy.action_kn : record.advisory.paddy.action_en;

  // #1. ನಾಳೆ ಕೂಲಿ ಬೇಕಾ? (48-Hour Coolie & Labour Booking Planner)
  const isRainRisk = lMax > 5.0 || exp >= 2.5;
  const coolieHtml = `
    <div class="decision-card ${isRainRisk ? "decision-hold" : "decision-safe"}" role="region" aria-label="Labour Booking Decision">
      <div class="decision-header">
        <span class="decision-tag">${currentLanguage === "kn" ? "ನಾಳೆ ಕೂಲಿ ಬೇಕಾ? (7 PM ನಿರ್ಧಾರ)" : "Tomorrow's Labour Booking (7 PM Decision)"}</span>
        <span class="decision-badge ${isRainRisk ? "badge-hold" : "badge-safe"}">
          ${isRainRisk ? (currentLanguage === "kn" ? "🚨 ಕೂಲಿ ಬೇಡ (HOLD)" : "🚨 HOLD LABOUR") : (currentLanguage === "kn" ? "✅ ಕೂಲಿ ಮುಂದುವರಿಸಿ (SAFE)" : "✅ SAFE TO BOOK")}
        </span>
      </div>
      <div class="decision-body">
        <p class="decision-main">
          ${
            isRainRisk
              ? currentLanguage === "kn"
                ? "ಸಂಜೆ/ಬೆಳಗ್ಗೆ ಮಳೆ ಸಾಧ್ಯತೆಯಿದೆ. ಕಳೆ ಕೀಳುವಿಕೆ ಅಥವಾ ಸಿಂಪಡಣೆ ಕೆಲಸಕ್ಕೆ ಕೂಲಿ ಬುಕ್ ಮಾಡಬೇಡಿ — <strong>₹800 ವರೆಗೆ ಕೂಲಿ ಹಣ ಉಳಿಸಿ</strong>."
                : "Rain risk expected during working hours. Avoid booking labour for weeding or spraying — <strong>save ~₹800 in wasted wages</strong>."
              : currentLanguage === "kn"
                ? "ಒಣ ಹವೆ / ಅನುಕೂಲಕರ ಹವಾಮಾನ. ಕಳೆ ಕೀಳುವಿಕೆ, ಗೊಬ್ಬರ ಹಾಗೂ ಸಿಂಪಡಣೆ ಕೆಲಸಕ್ಕೆ ಕೂಲಿಗಳನ್ನು ನಿರಾತಂಕವಾಗಿ ಬುಕ್ ಮಾಡಬಹುದು."
                : "Dry and favorable weather. Safe to contract agricultural labour for spraying, weeding, and intercultural operations."
          }
        </p>
      </div>
    </div>
  `;

  // #2. ಮಳೆ ನಂತರ ಕೀಟ ಎಚ್ಚರಿಕೆ (Post-Rain 48-Hour Blast & Pest Warning)
  const isPestRisk = lMax >= 15.0 || exp >= 10.0;
  const pestHtml = `
    <div class="pest-card ${isPestRisk ? "pest-alert" : "pest-low"}" role="region" aria-label="Post-rain pest advisory">
      <div class="pest-header">
        <span class="pest-icon">${isPestRisk ? "🍄" : "🛡️"}</span>
        <strong>${currentLanguage === "kn" ? "ಮಳೆ ನಂತರ ಕೀಟ/ರೋಗ ಎಚ್ಚರಿಕೆ (48h Protocol)" : "Post-Rain Pest & Blast Alert (48h Protocol)"}</strong>
        <span class="pest-level ${isPestRisk ? "level-high" : "level-low"}">
          ${isPestRisk ? (currentLanguage === "kn" ? "ತೀವ್ರ ನಿಗಾ" : "HIGH RISK") : (currentLanguage === "kn" ? "ಕಡಿಮೆ ಬಾಧೆ" : "LOW RISK")}
        </span>
      </div>
      <p class="pest-text">
        ${
          isPestRisk
            ? currentLanguage === "kn"
              ? "ಮಳೆ ನಿಂತ 48 ಗಂಟೆಗಳಲ್ಲಿ ಎಲೆ ಚುಕ್ಕೆ ಮತ್ತು <strong>ರಾಗಿ/ಭತ್ತದ ಬೆಂಕಿ ರೋಗ (Blast)</strong> ಹರಡುವ ಅಪಾಯವಿದೆ. ಮುಂಜಾಗ್ರತೆಯಾಗಿ ಜೈವಿಕ ಶಿಲೀಂಧ್ರನಾಶಕ <em>ಸೂಡೋಮೊನಾಸ್ (Pseudomonas 10g/L)</em> ಅಥವಾ <em>ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ 75% WP (0.6g/L)</em> ಔಷಧಿಯನ್ನು ಲಭ್ಯವಿಟ್ಟುಕೊಳ್ಳಿ."
              : "High canopy wetness will favor <strong>Ragi & Paddy Blast (Pyricularia oryzae)</strong> within 48h after rain. Keep bio-agent <em>Pseudomonas fluorescens (10g/L)</em> or <em>Tricyclazole 75% WP (0.6g/L)</em> ready for prophylactic spray."
            : currentLanguage === "kn"
              ? "ಪ್ರಸ್ತುತ ಹವಾಮಾನದಲ್ಲಿ ಕೀಟ ಮತ್ತು ಶಿಲೀಂಧ್ರ ಬಾಧೆ ಕಡಿಮೆ. ಸಾಮಾನ್ಯ ಕ್ಷೇತ್ರ ವೀಕ್ಷಣೆ ಮುಂದುವರಿಸಿ."
              : "Current micro-climate indicates low pest and fungal pressure. Continue routine crop surveillance."
        }
      </p>
    </div>
  `;

  // Grounded Agro-Economics
  const showEconomics = lMax > 10.0 || exp >= 15.0;
  const economicsHtml = showEconomics
    ? `
      <div class="economics-card" role="note" aria-label="Economic impact">
        <div class="economics-icon">💰</div>
        <div class="economics-body">
          <div class="economics-title">${currentLanguage === "kn" ? "ರೈತರ ಉಳಿತಾಯ (Protected Farm Input)" : "Smallholder Input Protected"}</div>
          <p class="economics-text">
            ${
              currentLanguage === "kn"
                ? "ಜೋರು ಮಳೆಯ ಮುನ್ಸೂಚನೆ ಇರುವಾಗ ರಸಗೊಬ್ಬರ ಹಾಕುವುದನ್ನು ಮುಂದೂಡುವುದರಿಂದ ಎಕರೆಗೆ ಸುಮಾರು <strong>₹700–₹1,200</strong> ಉಳಿತಾಯವಾಗುತ್ತದೆ (೧ ಚೀಲ ಡಿಎಪಿಗೆ ಸಮಾನ)."
                : "Postponing fertilizer before heavy rain prevents nitrogen leaching, protecting ~<strong>₹700–₹1,200 per acre</strong> (equivalent to 1 bag of DAP)."
            }
          </p>
          <span class="economics-source">${currentLanguage === "kn" ? "ಮೂಲ: ಕೃಷಿ ವೆಚ್ಚ ಮತ್ತು ಬೆಲೆ ಆಯೋಗ (DES) ಮಾನದಂಡ" : "Source: Directorate of Economics & Statistics (DES) Cultivation Benchmark"}</span>
        </div>
      </div>
    `
    : "";

  container.innerHTML = `
    <article class="forecast-card">
      <div class="forecast-header">
        <div>
          <h3 class="panchayat-title">${record.panchayat_name}</h3>
          <div class="panchayat-meta">
            LGD Code: <strong>${record.lgd_code}</strong> • District: ${record.district} • Forecast Date: ${record.forecast_date}
          </div>
        </div>
        <div class="rainfall-badge-container">
          <div class="rain-expected">${exp.toFixed(1)}<span class="rain-unit">mm</span></div>
          <span class="intensity-pill ${intensity.class}">${intensity.label}</span>
        </div>
      </div>

      <!-- Decision Trigger #1: 48-Hour Labour / Coolie Booking Planner -->
      ${coolieHtml}

      <!-- CQR 90% Calibrated Uncertainty Section -->
      <div class="cqr-box">
        <div class="cqr-header">
          <span>90% Calibrated Prediction Interval (CQR)</span>
          <span>Coverage Verified</span>
        </div>
        <div class="cqr-range">
          Likely Range: ${lMin.toFixed(1)} mm – ${lMax.toFixed(1)} mm
        </div>
        <div class="cqr-note">
          ${record.rainfall_mm.empirical_coverage} • Evaluated with clip-at-zero finite-sample correction.
        </div>
      </div>

      <!-- Bilingual Agro-Advisories -->
      <div class="advisories-grid">
        <div class="advisory-card">
          <div class="advisory-header">
            <span class="crop-name">🌱 Ragi (Finger Millet / ರಾಗಿ)</span>
            <span class="stage-tag">${record.advisory.ragi.stage}</span>
          </div>
          <p class="advisory-text" lang="${currentLanguage}">${ragiAdv}</p>
        </div>

        <div class="advisory-card">
          <div class="advisory-header">
            <span class="crop-name">🌾 Paddy (Rice / ಭತ್ತ)</span>
            <span class="stage-tag">${record.advisory.paddy.stage}</span>
          </div>
          <p class="advisory-text" lang="${currentLanguage}">${paddyAdv}</p>
        </div>
      </div>

      <!-- Decision Trigger #2: Post-Rain 48-Hour Pest & Blast Warning -->
      ${pestHtml}

      ${economicsHtml}

      <!-- Field Actions: Krishi Sakhi WhatsApp Broadcast & Voice Assistant & Chalkboard Mode -->
      <div class="field-actions-bar">
        <button id="btn-share-whatsapp" class="btn-action btn-whatsapp" title="Share forecast to WhatsApp" aria-label="Share forecast to WhatsApp">
          <span class="btn-action-icon">💬</span>
          <span>${currentLanguage === "kn" ? "ವಾಟ್ಸಾಪ್‌ನಲ್ಲಿ ಹಂಚಿಕೊಳ್ಳಿ" : "Share on WhatsApp"}</span>
        </button>

        <button id="btn-voice" class="btn-action btn-voice hidden" title="Listen to advisory" aria-label="Listen to advisory">
          <span class="btn-action-icon">🔊</span>
          <span id="voice-btn-text">${currentLanguage === "kn" ? "ಕೇಳಿ" : "Listen"}</span>
        </button>

        <button id="btn-katte-mode" class="btn-action btn-katte" title="Temple / Dairy Chalkboard Display" aria-label="Temple Chalkboard Display">
          <span class="btn-action-icon">📋</span>
          <span>${currentLanguage === "kn" ? "ಕಟ್ಟೆ ಚೀಟಿ (Chalkboard)" : "Chalkboard Mode"}</span>
        </button>
      </div>

      <!-- #4: ಗುಡಿ ಕಟ್ಟೆ ಚೀಟಿ (Chalkboard Modal for Non-Phone Farmers) -->
      <div id="katte-overlay" class="katte-overlay hidden" role="dialog" aria-modal="true">
        <div class="katte-board">
          <div class="katte-top">
            <span>${record.panchayat_name}</span>
            <span>${record.forecast_date}</span>
          </div>
          <div class="katte-symbol ${isRainRisk ? "katte-x" : "katte-check"}">
            ${isRainRisk ? "✕" : "✓"}
          </div>
          <div class="katte-action">
            ${
              isRainRisk
                ? currentLanguage === "kn"
                  ? "ಸಿಂಪಡಣೆ / ಕೂಲಿ ಬೇಡ (HOLD)"
                  : "NO SPRAY / HOLD LABOUR"
                : currentLanguage === "kn"
                  ? "ಕೆಲಸ ಮುಂದುವರಿಸಿ (PROCEED)"
                  : "SAFE FOR FIELD WORK"
            }
          </div>
          <div class="katte-rain-val">
            ${exp.toFixed(1)} mm (${lMin.toFixed(1)}–${lMax.toFixed(1)} mm)
          </div>
          <div class="katte-note">
            ${
              currentLanguage === "kn"
                ? "ದೇವಸ್ಥಾನದ ಕಟ್ಟೆ ಅಥವಾ ಹಾಲಿನ ಡೈರಿ ಬೋರ್ಡ್ ಮೇಲೆ ಸೀಮೆಸುಣ್ಣದಿಂದ ಬರೆಯಲು"
                : "Chalkboard template for village dairy / temple wall"
            }
          </div>
          <button id="btn-close-katte" class="btn-close-katte">✕ ${currentLanguage === "kn" ? "ಮುಚ್ಚಿ" : "Close"}</button>
        </div>
      </div>
    </article>
  `;

  // Attach Action Handlers
  const shareBtn = container.querySelector("#btn-share-whatsapp");
  if (shareBtn) {
    shareBtn.onclick = () => broadcastToWhatsApp(record);
  }

  const voiceBtn = container.querySelector("#btn-voice");
  if (voiceBtn) {
    voiceBtn.onclick = () => playVoiceAdvisory(record);
    checkVoiceAvailability(voiceBtn);
  }

  const katteBtn = container.querySelector("#btn-katte-mode");
  const katteOverlay = container.querySelector("#katte-overlay");
  const katteClose = container.querySelector("#btn-close-katte");
  if (katteBtn && katteOverlay && katteClose) {
    katteBtn.onclick = () => katteOverlay.classList.remove("hidden");
    katteClose.onclick = () => katteOverlay.classList.add("hidden");
    katteOverlay.onclick = e => {
      if (e.target === katteOverlay) katteOverlay.classList.add("hidden");
    };
  }
}

// -------------------------------------------------------------
// Krishi Sakhi WhatsApp Community Broadcaster (0% Risk Universal Link)
// -------------------------------------------------------------
function broadcastToWhatsApp(record) {
  const exp = record.rainfall_mm.expected;
  const lMin = record.rainfall_mm.likely_min;
  const lMax = record.rainfall_mm.likely_max;
  const intensity = getIntensityLabel(exp);

  let message = "";
  if (currentLanguage === "kn") {
    const alertLine = lMax > 10.0 ? "⚠️ ಎಚ್ಚರಿಕೆ: ಸಂಜೆ ಜೋರು ಮಳೆ ಸಾಧ್ಯತೆ ಇದೆ!" : "✅ ಸಾಮಾನ್ಯ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ";
    const econLine = lMax > 10.0 ? "\n💰 ಸಲಹೆ: ಗೊಬ್ಬರ ವ್ಯರ್ಥವಾಗುವುದನ್ನು ತಪ್ಪಿಸಿ (ಎಕರೆಗೆ ~₹700-1200 ಉಳಿತಾಯ)." : "";
    message =
      `🌾 *ಗ್ರಾಮ ಪಂಚಾಯತ್: ${record.panchayat_name}* (ಮಂಡ್ಯ ಜಿಲ್ಲೆ)\n` +
      `📅 ದಿನಾಂಕ: ${record.forecast_date}\n\n` +
      `🌧️ *ಮಳೆ ಮುನ್ಸೂಚನೆ:* ${intensity.label} (${exp.toFixed(1)} mm)\n` +
      `📊 *ಸಂಭಾವ್ಯ ವ್ಯಾಪ್ತಿ:* ${lMin.toFixed(1)} mm – ${lMax.toFixed(1)} mm\n` +
      `${alertLine}\n\n` +
      `🌱 *ರಾಗಿ ಬೆಳೆ ಸಲಹೆ:* ${record.advisory.ragi.action_kn}\n` +
      `🌾 *ಭತ್ತದ ಬೆಳೆ ಸಲಹೆ:* ${record.advisory.paddy.action_kn}\n` +
      `${econLine}\n` +
      `🔗 *ಮಂಡ್ಯ ಕೃಷಿ ಹವಾಮಾನ ಸೇವೆ*`;
  } else {
    const alertLine = lMax > 10.0 ? "⚠️ Alert: Evening heavy rainfall burst likely!" : "✅ Normal agricultural conditions";
    const econLine = lMax > 10.0 ? "\n💰 Input Notice: Postponing fertilizer protects ~₹700–1,200/acre." : "";
    message =
      `🌾 *Gram Panchayat: ${record.panchayat_name}* (Mandya District)\n` +
      `📅 Date: ${record.forecast_date}\n\n` +
      `🌧️ *Rainfall Forecast:* ${intensity.label} (${exp.toFixed(1)} mm)\n` +
      `📊 *CQR 90% Likely Range:* ${lMin.toFixed(1)} mm – ${lMax.toFixed(1)} mm\n` +
      `${alertLine}\n\n` +
      `🌱 *Ragi Advisory:* ${record.advisory.ragi.action_en}\n` +
      `🌾 *Paddy Advisory:* ${record.advisory.paddy.action_en}\n` +
      `${econLine}\n` +
      `🔗 *Mandya Agro-Weather Service*`;
  }

  const waUrl = `https://wa.me/?text=${encodeURIComponent(message)}`;
  if (navigator.share) {
    navigator
      .share({ title: `Weather Advisory - ${record.panchayat_name}`, text: message })
      .catch(() => window.open(waUrl, "_blank", "noopener,noreferrer"));
  } else {
    window.open(waUrl, "_blank", "noopener,noreferrer");
  }
}

// -------------------------------------------------------------
// Voice Assistant with "Ghost Button" Local-Service Guard
// -------------------------------------------------------------
function checkVoiceAvailability(btnElement) {
  if (!btnElement || !("speechSynthesis" in window)) {
    if (btnElement) btnElement.classList.add("hidden");
    return;
  }

  const voices = window.speechSynthesis.getVoices();
  const hasLocalKn = voices.some(
    v => v.localService && (v.lang.toLowerCase().includes("kn") || v.lang.toLowerCase().includes("kan"))
  );
  const hasEn = voices.some(v => v.lang.toLowerCase().includes("en"));

  if (currentLanguage === "kn") {
    // Only show if actual local Kannada voice exists to avoid English phoneme distortion
    if (hasLocalKn) {
      btnElement.classList.remove("hidden");
    } else {
      btnElement.classList.add("hidden");
    }
  } else {
    if (hasEn) {
      btnElement.classList.remove("hidden");
    } else {
      btnElement.classList.add("hidden");
    }
  }
}

function playVoiceAdvisory(record) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel(); // Stop ongoing speech

  const exp = record.rainfall_mm.expected;
  const lMax = record.rainfall_mm.likely_max;
  const voices = window.speechSynthesis.getVoices();

  let textToSpeak = "";
  let voiceToUse = null;

  if (currentLanguage === "kn") {
    voiceToUse = voices.find(
      v => v.localService && (v.lang.toLowerCase().includes("kn") || v.lang.toLowerCase().includes("kan"))
    );
    if (!voiceToUse) return; // Silent guard

    // Natural 2-sentence colloquial copy (no technical jargon or raw decimals)
    if (lMax > 10.0) {
      textToSpeak = `ಇಂದು ${record.panchayat_name}ದಲ್ಲಿ ಸಾಧಾರಣ ಮಳೆ ನಿರೀಕ್ಷೆ ಇದೆ. ಸಂಜೆ ಜೋರು ಮಳೆ ಸಾಧ್ಯತೆ ಇರುವುದರಿಂದ ರಾಗಿ ಬೆಳೆಗೆ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.`;
    } else if (exp >= 2.5) {
      textToSpeak = `ಇಂದು ${record.panchayat_name}ದಲ್ಲಿ ಹಗುರ ಮಳೆ ಬರಬಹುದು. ಕೃಷಿ ಕೆಲಸಗಳನ್ನು ಮುಂದುವರಿಸಬಹುದು.`;
    } else {
      textToSpeak = `ಇಂದು ${record.panchayat_name}ದಲ್ಲಿ ಒಣ ಹವೆ ಇರುತ್ತದೆ. ಅಗತ್ಯವಿದ್ದರೆ ನೀರಾವರಿ ಒದಗಿಸಬಹುದು.`;
    }
  } else {
    voiceToUse = voices.find(v => v.lang.toLowerCase().includes("en")) || null;
    if (lMax > 10.0) {
      textToSpeak = `Moderate rain expected in ${record.panchayat_name}. Heavy burst likely by evening. Please postpone fertilizer application.`;
    } else if (exp >= 2.5) {
      textToSpeak = `Light rain expected in ${record.panchayat_name}. Field operations can safely proceed.`;
    } else {
      textToSpeak = `Dry weather expected in ${record.panchayat_name}. Normal irrigation can continue.`;
    }
  }

  const utterance = new SpeechSynthesisUtterance(textToSpeak);
  if (voiceToUse) utterance.voice = voiceToUse;
  utterance.rate = 0.9; // 10% slower for field clarity
  utterance.pitch = 1.0;

  const btnText = document.querySelector("#voice-btn-text");
  if (btnText) btnText.textContent = currentLanguage === "kn" ? "ಪ್ಲೇ ಆಗುತ್ತಿದೆ…" : "Playing…";

  utterance.onend = () => {
    if (btnText) btnText.textContent = currentLanguage === "kn" ? "ಕೇಳಿ" : "Listen";
  };
  utterance.onerror = () => {
    if (btnText) btnText.textContent = currentLanguage === "kn" ? "ಕೇಳಿ" : "Listen";
  };

  window.speechSynthesis.speak(utterance);
}

// Render Choropleth Map
async function renderMap(records) {
  const status = document.querySelector("#map-status");
  const svg = document.querySelector("#map");
  const tooltip = document.querySelector("#map-tooltip");

  try {
    const res = await fetch("/mandya_simplified.topojson");
    if (!res.ok) throw new Error("Could not load TopoJSON map boundary");
    const topology = await res.json();

    const collection = Object.values(topology.objects).find(obj => obj.type === "GeometryCollection");
    if (!collection || !topology.transform) throw new Error("Invalid TopoJSON structure");

    const rainLookup = new Map(records.map(r => [String(r.lgd_code), r.rainfall_mm.expected]));
    const nameLookup = new Map(records.map(r => [String(r.lgd_code), r.panchayat_name]));

    const geometries = collection.geometries.filter(g => g.type === "Polygon" || g.type === "MultiPolygon");
    const allArcs = geometries.flatMap(g => (g.type === "Polygon" ? g.arcs : g.arcs.flat()));
    const points = allArcs.flatMap(ring => ring.flatMap(idx => decodeArc(topology, idx)));

    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const pad = 0.02;
    const width = (maxX - minX) * (1 + pad * 2);
    const height = (maxY - minY) * (1 + pad * 2);
    const viewBoxX = minX - (maxX - minX) * pad;
    const viewBoxY = -maxY - (maxY - minY) * pad;

    initialViewBox = { x: viewBoxX, y: viewBoxY, width, height };
    applyViewBox(svg, initialViewBox);

    svg.replaceChildren();
    mapPaths = new Map();

    geometries.forEach(geometry => {
      const code = String(geometry.properties.gpcode || geometry.properties.lgd_code || "").trim();
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathStringFor(topology, geometry.type === "Polygon" ? geometry.arcs : geometry.arcs.flat()));

      const expectedMm = rainLookup.get(code) || 0;
      path.style.fill = getRainColor(expectedMm);
      path.dataset.lgdCode = code;

      // Click event
      path.addEventListener("click", () => {
        const match = records.find(r => String(r.lgd_code) === code);
        if (match) {
          const select = document.querySelector("#panchayat-select");
          select.value = code;
          renderForecastDetails(match);
        }
      });

      // Hover Tooltip
      path.addEventListener("mouseenter", e => {
        const pName = nameLookup.get(code) || geometry.properties.gpname || `GP ${code}`;
        tooltip.textContent = `${pName}: ${expectedMm.toFixed(1)} mm`;
        tooltip.classList.remove("hidden");
      });

      path.addEventListener("mousemove", e => {
        const rect = svg.getBoundingClientRect();
        tooltip.style.left = `${e.clientX - rect.left}px`;
        tooltip.style.top = `${e.clientY - rect.top}px`;
      });

      path.addEventListener("mouseleave", () => {
        tooltip.classList.add("hidden");
      });

      svg.append(path);
      mapPaths.set(code, path);
    });

    status.textContent = `${geometries.length} Mandya panchayats loaded with downscaled choropleth.`;
  } catch (err) {
    status.textContent = "Map boundary rendering error. Please use panchayat dropdown.";
    console.error(err);
  }
}

function applyViewBox(svg, vb) {
  svg.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.width / currentZoom} ${vb.height / currentZoom}`);
}

// Setup Zoom/Pan Controls
function setupMapControls() {
  const svg = document.querySelector("#map");
  document.querySelector("#zoom-in").addEventListener("click", () => {
    if (currentZoom < 3.5) {
      currentZoom += 0.35;
      if (initialViewBox) applyViewBox(svg, initialViewBox);
    }
  });

  document.querySelector("#zoom-out").addEventListener("click", () => {
    if (currentZoom > 0.8) {
      currentZoom -= 0.35;
      if (initialViewBox) applyViewBox(svg, initialViewBox);
    }
  });

  document.querySelector("#zoom-reset").addEventListener("click", () => {
    currentZoom = 1.0;
    if (initialViewBox) applyViewBox(svg, initialViewBox);
  });
}

// Update District Summary Stats Bar
function updateStatsBar(records) {
  if (!records.length) return;
  const rains = records.map(r => r.rainfall_mm.expected);
  const avg = rains.reduce((a, b) => a + b, 0) / rains.length;
  const max = Math.max(...rains);
  const wetCount = rains.filter(mm => mm >= 2.5).length;

  document.querySelector("#stat-total").textContent = records.length;
  document.querySelector("#stat-avg").textContent = `${avg.toFixed(1)} mm`;
  document.querySelector("#stat-max").textContent = `${max.toFixed(1)} mm`;
  document.querySelector("#stat-wet").textContent = `${wetCount}/${records.length} Wet`;
}

// Load Application Data
async function loadData() {
  let records = [];
  let isCachedMode = false;

  try {
    const res = await fetch("/api/forecasts");
    if (!res.ok) throw new Error("Network API unavailable");
    records = await res.json();
    await saveAllToDb(records);
  } catch {
    records = await getAllFromDb();
    isCachedMode = true;
  }

  currentRecords = records;
  const isOffline = isCachedMode || !navigator.onLine;

  // Status and Offline Banner Management
  const banner = document.querySelector("#offline-banner");
  const bannerText = document.querySelector("#offline-banner-text");
  const syncBadge = document.querySelector("#sync-badge");
  const syncStatus = document.querySelector("#sync-status");

  const timestamp = records[0]?.timestamp_utc
    ? new Date(records[0].timestamp_utc).toLocaleString()
    : "recent sync";

  if (isOffline) {
    banner.classList.remove("hidden");
    bannerText.textContent = `⚠️ Offline: Viewing cached forecast from ${timestamp}. Local generation not supported.`;
    syncBadge.classList.add("offline-mode");
    syncStatus.textContent = `Offline: Cached (${records.length} GPs)`;
  } else {
    banner.classList.add("hidden");
    syncBadge.classList.remove("offline-mode");
    syncStatus.textContent = `Last synced: ${timestamp} (${records.length} GPs)`;
  }

  // Populate Dropdown
  const select = document.querySelector("#panchayat-select");
  select.replaceChildren(
    ...records.map(r => new Option(`${r.panchayat_name} (${r.rainfall_mm.expected.toFixed(1)} mm)`, r.lgd_code))
  );

  select.onchange = () => {
    const match = records.find(r => r.lgd_code === select.value);
    if (match) renderForecastDetails(match);
  };

  // Language Toggles
  document.querySelectorAll(".lang-btn").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".lang-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentLanguage = btn.dataset.lang;
      const current = records.find(r => r.lgd_code === (selectedLgdCode || select.value));
      if (current) renderForecastDetails(current);
    };
  });

  updateStatsBar(records);
  await renderMap(records);

  if (records.length) {
    const initial = records.find(r => r.lgd_code === selectedLgdCode) || records[0];
    select.value = initial.lgd_code;
    renderForecastDetails(initial);
  }
}

// Setup Jury / Science Drawer Toggle
function setupJuryDrawer() {
  const btn = document.querySelector("#btn-jury-mode");
  const drawer = document.querySelector("#jury-drawer");
  if (!btn || !drawer) return;

  btn.addEventListener("click", () => {
    const isHidden = drawer.classList.contains("hidden");
    drawer.classList.toggle("hidden", !isHidden);
    btn.setAttribute("aria-expanded", String(isHidden));
    const icon = btn.querySelector(".toggle-icon");
    if (icon) icon.textContent = isHidden ? "▲" : "▼";
  });
}

// Lifecycle Events
window.addEventListener("offline", loadData);
window.addEventListener("online", loadData);

if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = () => {
    const voiceBtn = document.querySelector("#btn-voice");
    if (voiceBtn) checkVoiceAvailability(voiceBtn);
  };
}

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/service-worker.js").catch(console.error);
}

setupMapControls();
setupJuryDrawer();
loadData();
