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
    </article>
  `;
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

// Lifecycle Events
window.addEventListener("offline", loadData);
window.addEventListener("online", loadData);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/service-worker.js").catch(console.error);
}

setupMapControls();
loadData();
