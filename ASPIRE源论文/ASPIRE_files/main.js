/**
 * ASPIRE project page
 */

document.addEventListener("DOMContentLoaded", () => {
  injectDynamicContent();
  initNavbar();
  initMethodVideoFullscreen();
  initCitationCopy();
});

function initCitationCopy() {
  const btn = document.getElementById("bibtex-copy");
  const code = document.getElementById("bibtex-code");
  if (!btn || !code) return;
  btn.addEventListener("click", () => {
    const text = code.textContent;
    const done = () => {
      btn.textContent = "Copied";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = "Copy";
        btn.classList.remove("copied");
      }, 1600);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
    } else {
      fallbackCopy(text, done);
    }
  });
}

function fallbackCopy(text, done) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); done(); } catch (e) { /* no-op */ }
  document.body.removeChild(ta);
}

window.addEventListener("load", () => {
  initAllCharts();
});

function injectDynamicContent() {
  const cfg = SITE_CONFIG;

  const titleEl = document.getElementById("paper-title");
  if (titleEl) {
    const parts = cfg.paperTitle.split(":");
    titleEl.innerHTML =
      parts.length === 2
        ? `<span class="accent">${parts[0].trim()}:</span> ${parts[1].trim()}`
        : cfg.paperTitle;
  }

  const introEl = document.getElementById("intro-text");
  if (introEl && cfg.introText) {
    introEl.innerHTML = renderIntroText(cfg.introText);
  }

  const teamEl = document.getElementById("nav-team");
  if (teamEl && cfg.teamName) {
    teamEl.textContent = cfg.teamName;
    if (cfg.teamUrl) teamEl.href = cfg.teamUrl;
  } else if (teamEl) {
    teamEl.style.display = "none";
  }

  document.querySelectorAll("[data-link]").forEach((el) => {
    const key = el.getAttribute("data-link");
    if (cfg.links[key]) {
      el.href = cfg.links[key];
      el.target = "_blank";
      el.rel = "noopener noreferrer";
    }
  });

  const authorsEl = document.getElementById("authors-list");
  if (authorsEl && cfg.authorHtml) {
    authorsEl.innerHTML = cfg.authorHtml;
  } else if (authorsEl && cfg.anonymous) {
    authorsEl.textContent = cfg.authors[0].name;
  } else if (authorsEl && cfg.authors.length) {
    authorsEl.textContent = cfg.authors.map((a) => a.name).join(", ");
  }
}

function renderIntroText(text) {
  let html = escapeHtml(text);
  const componentTerms = [
    { marker: "(1)", phrase: "robot execution engine" },
    { marker: "(2)", phrase: "skill library" },
    { marker: "(3)", phrase: "evolutionary search" },
  ];

  componentTerms.forEach(({ marker, phrase }) => {
    const pattern = new RegExp(`(${escapeRegExp(marker)}[\\s\\S]*?)\\b(${escapeRegExp(phrase)})\\b`, "i");
    html = html.replace(pattern, "$1<strong>$2</strong>");
  });

  return html;
}

function escapeHtml(value) {
  const entities = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  return String(value).replace(/[&<>"']/g, (char) => entities[char]);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function initNavbar() {
  const navbar = document.querySelector(".navbar");
  if (!navbar) return;
  // The left rail stays hidden over BOTH front-page heroes (the cover video
  // and the grid wall) and slides in once the grid hero is scrolled past.
  const anchor = document.getElementById("grid-hero") || document.getElementById("hero");
  const threshold = () => (anchor ? anchor.offsetTop + anchor.offsetHeight - 96 : 12);
  const update = () => {
    const docked = window.scrollY > threshold();
    navbar.classList.toggle("scrolled", docked);
    document.body.classList.toggle("nav-docked", docked);
  };
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

function initMethodVideoFullscreen() {
  const videos = Array.from(document.querySelectorAll("#method .component-media video"));
  if (!videos.length) return;

  let activeSource = null;
  const root = document.createElement("div");
  root.className = "method-video-modal";
  root.setAttribute("role", "dialog");
  root.setAttribute("aria-modal", "true");
  root.setAttribute("aria-label", "Expanded method video");
  root.innerHTML =
    '<button class="method-video-close" type="button" aria-label="Close expanded video">&times;</button>' +
    '<video class="method-video-full" controls playsinline></video>';
  document.body.appendChild(root);

  const fullVideo = root.querySelector(".method-video-full");
  const closeBtn = root.querySelector(".method-video-close");

  function open(source) {
    activeSource = source;
    source.pause();
    fullVideo.src = source.currentSrc || source.getAttribute("src") || "";
    fullVideo.poster = source.getAttribute("poster") || "";
    // Always begin the expanded video from the start, not wherever the
    // muted preview happened to be.
    fullVideo.currentTime = 0;
    fullVideo.muted = false;
    root.classList.add("open");
    document.body.style.overflow = "hidden";
    closeBtn.focus({ preventScroll: true });
    const start = () => {
      fullVideo.currentTime = 0;
      const play = fullVideo.play();
      if (play) play.catch(() => {});
    };
    // currentTime can't be set reliably until metadata is loaded; if it
    // isn't ready yet, reset once it is so playback truly starts at 0.
    if (fullVideo.readyState >= 1) start();
    else fullVideo.addEventListener("loadedmetadata", start, { once: true });
  }

  function close() {
    if (!root.classList.contains("open")) return;
    fullVideo.pause();
    // Restart the muted preview from the beginning so it keeps looping cleanly.
    if (activeSource) {
      activeSource.currentTime = 0;
      const play = activeSource.play();
      if (play) play.catch(() => {});
    }
    fullVideo.removeAttribute("src");
    fullVideo.load();
    root.classList.remove("open");
    document.body.style.overflow = "";
    activeSource = null;
  }

  videos.forEach((video) => {
    const frame = video.closest(".component-media");
    if (!frame) return;
    frame.classList.add("method-video-expandable");
    frame.setAttribute("role", "button");
    frame.setAttribute("tabindex", "0");
    frame.setAttribute("aria-label", "Expand method video");
    frame.addEventListener("click", () => open(video));
    frame.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      open(video);
    });
  });

  closeBtn.addEventListener("click", close);
  root.addEventListener("click", (event) => {
    if (event.target === root) close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
  });
}

/* ── Sidebar scroll-spy: highlight the section currently in view ── */
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    var links = Array.prototype.slice.call(document.querySelectorAll("[data-spy]"));
    if (!links.length) return;
    var sections = links
      .map(function (l) { return { id: l.dataset.spy, el: document.getElementById(l.dataset.spy) }; })
      .filter(function (s) { return s.el; });
    var current = null;
    function update() {
      var y = window.scrollY + window.innerHeight * 0.32;
      var active = sections.length ? sections[0].id : null;
      sections.forEach(function (s) { if (s.el.offsetTop <= y) active = s.id; });
      if (active === current) return;
      current = active;
      links.forEach(function (l) { l.classList.toggle("active", l.dataset.spy === active); });
    }
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  });
})();

/* ── Pause videos that aren't on screen (saves a lot of CPU/battery) ── */
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    if (!("IntersectionObserver" in window)) return;
    var vids = document.querySelectorAll(".component-media video, .reality-video, .pp-frame video, .method-figure-video, .overview-video video");
    if (!vids.length) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) { var p = v.play(); if (p) p.catch(function () {}); }
        else v.pause();
      });
    }, { threshold: 0.12 });
    vids.forEach(function (v) { io.observe(v); });
  });
})();
