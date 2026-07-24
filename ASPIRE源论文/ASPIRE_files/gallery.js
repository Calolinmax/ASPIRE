/**
 * ASPIRE task gallery.
 *
 * Renders one filtered before/after gallery. The page owns scrolling; the
 * gallery itself is not a nested scroll container.
 */
(function () {
  "use strict";

  const EXTRA_GALLERY_URL = "data/gallery_libero90.json?v=65";
  const VIDEO_VERSION = "v=7";
  const PAGE_SIZE = 12;
  // Phones (portrait or short landscape) show 3 tasks per page; desktop shows PAGE_SIZE.
  const isMobile = () => window.matchMedia("(max-width: 768px), (orientation: landscape) and (max-height: 600px)").matches;
  const isLandscapePhone = () => window.matchMedia("(orientation: landscape) and (max-height: 600px)").matches;
  // Portrait phone: 3/page · Landscape phone: 4/page · Desktop: PAGE_SIZE.
  const pageSize = () => (isLandscapePhone() ? 4 : isMobile() ? 3 : PAGE_SIZE);

  const FILTERS = [
    { key: "all", label: "All" },
    { key: "Real", label: "Real" },
    { key: "BEHAVIOR-1K", label: "BEHAVIOR-1K" },
    { key: "LIBERO", label: "LIBERO" },
    { key: "Robosuite", label: "Robosuite" },
  ];

  const V = (name) => `assets/videos/gallery/${name}.mp4`;
  const C = (name) => `assets/code/gallery/${name}.py`;
  const norm = (value) => String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const tokenizePythonLine = (line) => window.AspireSyntax.tokenizePythonLine(line);

  // Deterministic shuffle: a fixed seed means the "random" order is computed
  // once and stays identical on every load (no reshuffling per visit).
  const SHUFFLE_SEED = 0x5eed1e;
  function makeRng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffled(list, rng) {
    const a = list.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  const ROBOSUITE_TASKS = [
    { label: "Cube Lift", before: { v: V("cube_lift_before"), c: C("cube_lift_before") }, after: { v: V("cube_lift_after"), c: C("cube_lift_after") } },
    { label: "Cube Stack", before: { v: V("cube_stack_before"), c: C("cube_stack_before") }, after: { v: V("cube_stack_after"), c: C("cube_stack_after") } },
    { label: "Cube Restack", before: { v: V("cube_restack_before"), c: C("cube_restack_before") }, after: { v: V("cube_restack_after"), c: C("cube_restack_after") } },
    { label: "Two-Arm Handover", before: { v: V("handover_before"), c: C("handover_before") }, after: { v: V("handover_after"), c: C("handover_after") } },
    { label: "Two-Arm Lift", before: { v: V("two_arm_lift_before"), c: C("two_arm_lift_before") }, after: { v: "assets/videos/grid_robo_two_arm_lift.mp4", c: C("two_arm_lift_after") } },
    { label: "Spill Wipe", before: { v: V("spill_wipe_before"), c: C("spill_wipe_before") }, after: { v: V("spill_wipe_after"), c: C("spill_wipe_after") } },
    { label: "Nut Assembly", before: { v: V("nut_assembly_before"), c: C("nut_assembly_before") }, after: { v: V("nut_assembly_after"), c: C("nut_assembly_after") } },
  ];

  const MORE_DEMOS = [
    { label: "Open the drawer and stow the bowl", domain: "LIBERO", before: { v: "assets/videos/drawer_baseline_failure.mp4", rate: 3 }, after: { v: "assets/videos/drawer_fix_success.mp4", rate: 3 } },
    { label: "Pick up the soda can", domain: "BEHAVIOR-1K", before: { v: "assets/videos/b1k/soda_before.mp4", rate: 8 }, after: { v: "assets/videos/b1k/soda_after.mp4", rate: 8 } },
    { label: "Pour water into the cup", domain: "Real", before: { v: "assets/videos/yam_pour_water_failure.mp4", rate: 2 }, after: { v: "assets/videos/yam_pour_water.mp4", rate: 2 } },
    { label: "Place the bowl on the plate", domain: "LIBERO", before: { v: "assets/videos/bowl_baseline_failure.mp4" }, after: { v: "assets/videos/bowl_fix_success.mp4" } },
    { label: "Pick up the radio", domain: "BEHAVIOR-1K", before: { v: "assets/videos/b1k/radio_before.mp4", c: "assets/code/b1k/radio_before.py", rate: 8 }, after: { v: "assets/videos/b1k/radio_after.mp4", c: "assets/code/b1k/radio_after.py", rate: 8 } },
  ];

  // Real-world YAM rollouts (top-camera), baseline failure vs ASPIRE success.
  // Videos played at 5x; code.py is the program executed in each run.
  const REAL_TASKS = [
    { label: "Place the bowl on the plate",
      before: { v: "assets/videos/real/bowl_before.mp4", c: "assets/code/real/bowl_before.py", rate: 5 },
      after:  { v: "assets/videos/real/bowl_after.mp4",  c: "assets/code/real/bowl_after.py",  rate: 5 } },
    { label: "Pick up the soda can",
      before: { v: "assets/videos/real/can_before.mp4", c: "assets/code/real/can_before.py", rate: 5 },
      after:  { v: "assets/videos/real/can_after.mp4",  c: "assets/code/real/can_after.py",  rate: 5 } },
    { label: "Open and push the drawer",
      before: { v: "assets/videos/real/drawer_before.mp4", c: "assets/code/real/drawer_before.py", rate: 5 },
      after:  { v: "assets/videos/real/drawer_after.mp4",  c: "assets/code/real/drawer_after.py",  rate: 5 } },
  ];

  let tasks = [];
  let activeFilter = "all";
  let activeQuery = "";
  let currentPage = 1;
  let videoObserver = null;
  let modal = null;
  const codeCache = {};

  function renderCode(text) {
    const lines = String(text || "Code unavailable.").replace(/\s+$/, "").split("\n");
    return lines.map((line) => `<span class="gcv-line">${tokenizePythonLine(line)}</span>`).join("");
  }

  async function fetchCode(url) {
    if (codeCache[url]) return codeCache[url];
    let text = "";
    try {
      const response = await fetch(url);
      text = response.ok ? await response.text() : "";
    } catch (_) {}
    codeCache[url] = text;
    return text;
  }

  function ensureCodeModal() {
    if (modal) return modal;
    const root = document.createElement("div");
    root.className = "gcv";
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.innerHTML =
      '<div class="gcv-backdrop"></div>' +
      '<div class="gcv-window">' +
      '  <div class="gcv-bar"><span class="gcv-title"></span><button class="gcv-close" type="button" aria-label="Close">Esc</button></div>' +
      '  <div class="gcv-panes"></div>' +
      '</div>';
    document.body.appendChild(root);

    const close = () => {
      root.classList.remove("open");
      document.body.style.overflow = "";
    };
    root.querySelector(".gcv-backdrop").addEventListener("click", close);
    root.querySelector(".gcv-close").addEventListener("click", close);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && root.classList.contains("open")) close();
    });

    modal = {
      root,
      title: root.querySelector(".gcv-title"),
      panes: root.querySelector(".gcv-panes"),
    };
    return modal;
  }

  async function openCode(task) {
    const viewer = ensureCodeModal();
    const sides = [];
    if (task.before && task.before.c) sides.push({ label: task.before.codeLabel || "Baseline", cls: "gcv-fail", url: task.before.c });
    if (task.after && task.after.c) sides.push({ label: task.after.codeLabel || "ASPIRE", cls: "gcv-pass", url: task.after.c });

    viewer.title.textContent = task.label;
    viewer.panes.className = `gcv-panes${sides.length === 2 ? " two" : ""}`;
    viewer.panes.innerHTML = sides
      .map((side) => (
        `<div class="gcv-pane"><div class="gcv-pane-head ${side.cls}">${side.label}</div>` +
        `<pre class="gcv-pre"><code class="gcv-code" data-url="${side.url}">loading...</code></pre></div>`
      ))
      .join("");
    viewer.root.classList.add("open");
    document.body.style.overflow = "hidden";

    for (const codeEl of viewer.panes.querySelectorAll(".gcv-code")) {
      codeEl.innerHTML = renderCode(await fetchCode(codeEl.dataset.url));
    }
  }

  function versionedVideoUrl(url) {
    return url.includes("?") ? `${url}&${VIDEO_VERSION}` : `${url}?${VIDEO_VERSION}`;
  }

  function renderClip(side, label, className) {
    if (!side) return "";
    const rate = side.rate ? ` data-rate="${side.rate}"` : "";
    return (
      `<div class="pair-clip"><video src="${versionedVideoUrl(side.v)}"${rate} muted loop playsinline preload="none"></video>` +
      `<span class="ba-tag ${className}">${label}</span></div>`
    );
  }

  function createCard(task) {
    const hasBothSides = task.before && task.after;
    const hasCode = (task.before && task.before.c) || (task.after && task.after.c);
    const clips = hasBothSides
      ? renderClip(task.before, "Baseline", "ba-tag-fail") + renderClip(task.after, "ASPIRE", "ba-tag-pass")
      : renderClip(task.after || task.before, task.after ? "ASPIRE" : "Baseline", task.after ? "ba-tag-pass" : "ba-tag-fail");

    const figure = document.createElement("figure");
    figure.className = "ba-pair";
    const videoClass = `pair-videos${hasBothSides ? "" : " gallery-single"}${task.wide ? " gallery-wide" : ""}`;
    const viewCode = hasCode
      ? '<button class="gallery-viewcode" type="button" aria-label="View code">' +
        '<svg class="eye" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">' +
        '<path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg><span>View code</span></button>'
      : "";

    figure.innerHTML = `<div class="${videoClass}">${clips}${viewCode}</div><figcaption>${task.label}</figcaption>`;
    if (hasCode) figure.querySelector(".gallery-viewcode").addEventListener("click", () => openCode(task));
    return figure;
  }

  function taskSearchText(task) {
    return norm([task.label, task.domain, task.group, task.room, task.search].join(" "));
  }

  function visibleTasks() {
    return tasks.filter((task) => {
      if (activeFilter !== "all" && task.domain !== activeFilter) return false;
      return !activeQuery || taskSearchText(task).includes(activeQuery);
    });
  }

  function syncControls(shown, totalPages) {
    const countEl = document.getElementById("gallery-count");
    const pageEl = document.getElementById("gallery-page");
    const prevEl = document.getElementById("gallery-prev");
    const nextEl = document.getElementById("gallery-next");

    if (countEl) countEl.textContent = `${shown}/${tasks.length}`;
    if (pageEl) pageEl.textContent = totalPages > 1 ? `Page ${currentPage} of ${totalPages}` : "Page 1 of 1";
    if (prevEl) prevEl.disabled = currentPage <= 1;
    if (nextEl) nextEl.disabled = currentPage >= totalPages;

    document.querySelectorAll(".gallery-filter").forEach((button) => {
      button.classList.toggle("active", button.dataset.filter === activeFilter);
    });
  }

  function observeVideos(grid) {
    if (videoObserver) videoObserver.disconnect();
    if (!("IntersectionObserver" in window)) return;
    videoObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting) {
          const promise = video.play();
          if (promise) promise.catch(() => {});
        } else {
          video.pause();
        }
      });
    }, { threshold: 0.15 });
    grid.querySelectorAll(".pair-clip video").forEach((video) => videoObserver.observe(video));
  }

  function applyPlaybackRates(grid) {
    grid.querySelectorAll(".pair-clip video").forEach((video) => {
      const rate = video.dataset.rate ? parseFloat(video.dataset.rate) : 0;
      if (!rate) return;
      const apply = () => { video.playbackRate = rate; };
      apply();
      video.addEventListener("loadedmetadata", apply);
    });
  }

  function renderGallery() {
    const grid = document.getElementById("gallery-grid");
    if (!grid) return;
    const ps = pageSize();
    const shown = visibleTasks();
    const totalPages = Math.max(1, Math.ceil(shown.length / ps));
    currentPage = Math.min(currentPage, totalPages);
    const start = (currentPage - 1) * ps;
    const pageTasks = shown.slice(start, start + ps);

    grid.innerHTML = "";
    if (!shown.length) {
      grid.innerHTML = '<div class="gallery-empty">No demos match the current filters.</div>';
      currentPage = 1;
      syncControls(0, 1);
      return;
    }

    pageTasks.forEach((task) => grid.appendChild(createCard(task)));
    syncControls(shown.length, totalPages);
    applyPlaybackRates(grid);
    observeVideos(grid);
  }

  function initControls() {
    const filtersEl = document.getElementById("gallery-room-filters");
    const searchEl = document.getElementById("gallery-search");
    const paginationEl = document.getElementById("gallery-pagination");
    if (!filtersEl || !searchEl) return;

    filtersEl.innerHTML = FILTERS
      .map((filter) => `<button class="gallery-filter${filter.key === "all" ? " active" : ""}" type="button" data-filter="${filter.key}">${filter.label}</button>`)
      .join("");

    filtersEl.addEventListener("click", (event) => {
      const button = event.target.closest(".gallery-filter");
      if (!button) return;
      activeFilter = button.dataset.filter;
      currentPage = 1;
      renderGallery();
    });

    searchEl.addEventListener("input", () => {
      activeQuery = norm(searchEl.value);
      currentPage = 1;
      renderGallery();
    });

    if (paginationEl) {
      paginationEl.addEventListener("click", (event) => {
        const button = event.target.closest("[data-page-action]");
        if (!button || button.disabled) return;
        currentPage += button.dataset.pageAction === "next" ? 1 : -1;
        renderGallery();
        document.getElementById("gallery")?.scrollIntoView({ block: "start", behavior: "smooth" });
      });
    }
  }

  async function loadTasks() {
    const loadedTasks = ROBOSUITE_TASKS.map((task) => ({
      ...task,
      domain: "Robosuite",
      group: "Robosuite",
    }));

    try {
      const response = await fetch(EXTRA_GALLERY_URL);
      if (response.ok) {
        const grouped = await response.json();
        Object.entries(grouped).forEach(([group, groupTasks]) => {
          const room = group.replace(/^LIBERO\s+/, "");
          groupTasks.forEach((task) => {
            loadedTasks.push({
              ...task,
              domain: "LIBERO",
              group,
              room,
              search: norm([task.label, group, room, "LIBERO-90"].join(" ")),
            });
          });
        });
      }
    } catch (_) {}

    MORE_DEMOS.forEach((task) => {
      loadedTasks.push({
        ...task,
        group: "More demos",
        search: norm([task.label, task.domain, "More demos"].join(" ")),
      });
    });

    REAL_TASKS.forEach((task) => {
      loadedTasks.push({
        ...task,
        domain: "Real",
        group: "Real",
        search: norm([task.label, "Real World YAM"].join(" ")),
      });
    });

    // One-time deterministic shuffle. Each domain is shuffled (so LIBERO,
    // Robosuite, etc. don't show in load order), then for the "All" view the
    // real-robot rollouts are placed up front but scattered across the first
    // page so they don't read as one solid block at the very top.
    const rng = makeRng(SHUFFLE_SEED);
    const real = shuffled(loadedTasks.filter((t) => t.domain === "Real"), rng);
    const rest = shuffled(loadedTasks.filter((t) => t.domain !== "Real"), rng);

    // Reserve a few of the first page's slots for the real demos, scattered
    // within the leading portion (kept in their shuffled order).
    const front = Math.min(PAGE_SIZE, 8);
    const slots = new Set();
    while (slots.size < real.length && slots.size < front) {
      slots.add(Math.floor(rng() * front));
    }
    const realSlots = [...slots].sort((a, b) => a - b);

    const ordered = [];
    let ri = 0;
    let qi = 0;
    const total = real.length + rest.length;
    for (let i = 0; i < total; i++) {
      if (ri < realSlots.length && realSlots[ri] === i) ordered.push(real[ri++]);
      else ordered.push(rest[qi++]);
    }

    tasks = ordered;
  }

  async function init() {
    const grid = document.getElementById("gallery-grid");
    if (!grid) return;
    await loadTasks();
    initControls();
    renderGallery();

    // Re-paginate when crossing the mobile breakpoint (e.g., rotating the phone)
    let lastPS = pageSize();
    window.addEventListener("resize", () => {
      const ps = pageSize();
      if (ps === lastPS) return;
      lastPS = ps;
      currentPage = 1;
      renderGallery();
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
