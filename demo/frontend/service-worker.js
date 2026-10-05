const CACHE = "mandya-v3-pwa-v26";
const STATIC_ASSETS = [
  "/",
  "/index.html",
  "/styles.css",
  "/app.js",
  "/manifest.webmanifest",
  "/icon.svg",
  "/icon-192.png",
  "/icon-512.png",
  "/mandya_simplified.topojson",
  "/leaflet/leaflet.js",
  "/leaflet/leaflet.css",
  "/audio/ragi_veg_rain_kn.mp3",
  "/audio/ragi_harvest_rot_kn.mp3",
  "/audio/ragi_sow_dry_kn.mp3",
  "/audio/paddy_veg_rain_kn.mp3",
  "/audio/paddy_harvest_rot_kn.mp3",
  "/audio/dry_window_safe_kn.mp3",
  "/audio/heavy_cloudburst_kn.mp3"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);

  // Network-First with Cache Fallback for API forecast queries
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE).then(cache => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(async () => {
          const cached = await caches.match(event.request, { ignoreSearch: true });
          if (cached) {
            const headers = new Headers(cached.headers);
            headers.set("X-Cache-Fallback", "1");
            return new Response(cached.body, {
              status: cached.status,
              statusText: cached.statusText,
              headers
            });
          }
          return cached;
        })
    );
    return;
  }

  // Network-First with Cache Fallback for all assets (ensures immediate updates when online, 100% offline resilience when disconnected)
  event.respondWith(
    fetch(event.request)
      .then(response => {
        if (response.ok && event.request.method === "GET" && url.origin === self.location.origin) {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request, { ignoreSearch: true }))
  );
});
