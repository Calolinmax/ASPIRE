/**
 * Small page-wide interactions shared across sections.
 */
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", () => {
    initScrollReveal();
    initScrollProgress();
  });

  function initScrollReveal() {
    const selector = ".section-header, .figure-block, .chart-block, .figure-caption";
    const targets = Array.from(document.querySelectorAll(selector));
    if (!("IntersectionObserver" in window) || !targets.length) return;

    targets.forEach((el) => el.classList.add("reveal"));
    const observer = new IntersectionObserver((entries, obs) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        obs.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });

    targets.forEach((el) => observer.observe(el));
  }

  function initScrollProgress() {
    const bar = document.getElementById("scroll-progress");
    if (!bar) return;

    const update = () => {
      const root = document.documentElement;
      const max = root.scrollHeight - root.clientHeight;
      bar.style.width = `${max > 0 ? (root.scrollTop / max) * 100 : 0}%`;
    };

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  }
})();
