// Service Worker — Le Panetier Sion
// Versione aggiornata automaticamente ad ogni deploy
const CACHE = "panetier-v20260316";
const FILES = ["/", "/index.html", "/manifest.json"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES)));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  // Elimina tutte le versioni precedenti della cache
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);

  // Per index.html: network-first (sempre versione aggiornata)
  // Se la rete fallisce, usa la cache come fallback
  if (url.pathname === "/" || url.pathname === "/index.html") {
    e.respondWith(
      fetch(e.request)
        .then(res => {
          // Aggiorna la cache con la versione fresca
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
          return res;
        })
        .catch(() => caches.match("/index.html"))
    );
    return;
  }

  // Per tutti gli altri file: cache-first
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request).catch(() => caches.match("/index.html")))
  );
});
