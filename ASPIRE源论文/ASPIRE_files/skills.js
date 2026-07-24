/**
 * ASPIRE skill library gallery.
 */
(function () {
  "use strict";

  const DATA_URL = "data/skill_library.json?v=27";
  const DOMAIN_ORDER = ["BEHAVIOR-1K", "LIBERO", "Robosuite"];
  const SKILL_PAGE_SIZE = 5;
  // Phones (portrait or short landscape) show 3 skills per page; desktop shows 5.
  const isMobile = () => window.matchMedia("(max-width: 768px), (orientation: landscape) and (max-height: 600px)").matches;
  const skillPageSize = () => (isMobile() ? 3 : SKILL_PAGE_SIZE);
  const CARD_SUMMARIES = {
    "behavior-grasping": "Verify grasps visually; interleave arms, poses, and planners.",
    "behavior-interactive-policy": "Build observe-act-verify loops one replayed block at a time.",
    "behavior-navigation": "Verify base motion and retry approaches from multiple angles.",
    "behavior-perception-and-search": "Search, observe, reprompt, and recover from bad masks.",
    "behavior-radio-table-tasks": "Use table geometry to approach and grasp tabletop objects.",
    "behavior-search": "Move between views until the target is localized.",
    "behavior-time-budget": "Spend the 900s budget on the highest-value retries.",
    "libero-fix-loop-grasp": "Reusable grasp-lift-place code templates for LIBERO tasks.",
    "libero-fix-loop-localize": "SAM3 localization, disambiguation, and prompt conventions.",
    "libero-fix-loop-manipulation": "Drawer, knob, microwave, push, and contact-heavy patterns.",
    "libero-fix-loop-transport": "Top-down transport, waypointing, and placement control.",
    "libero-library-scaling-grasp": "Expanded LIBERO-90 grasp registry with offsets and thresholds.",
    "libero-library-scaling-localize": "Larger prompt and geometry registry for broader transfer.",
    "libero-library-scaling-manipulation": "Accumulated articulated-object parameters for transfer.",
    "libero-library-scaling-transport": "End-effector transit and placement patterns for safe motion.",
    "robosuite-fix-loop-grasp": "Top-down grasp-lift-place skeletons for Robosuite tasks.",
    "robosuite-fix-loop-localize": "SAM3 localization patterns for Robosuite objects.",
    "robosuite-fix-loop-transport": "Collision-aware transport paths for Robosuite objects.",
  };

  let skills = [];
  let activeDomain = "all";
  let activeQuery = "";
  let activeId = "";
  let activePage = 0;

  const norm = (value) => String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

  function esc(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  const IMPLICIT_CODE_PATTERN = new RegExp([
    String.raw`\.?[A-Za-z0-9_./-]+\.(?:py|md|json|ya?ml)(?:::[A-Za-z_][A-Za-z0-9_]*\([^)]{0,80}\))?`,
    String.raw`\b[A-Za-z_][A-Za-z0-9_]*\[[^\]]{1,40}\]`,
    String.raw`\b[A-Za-z_][A-Za-z0-9_]*\s*[+\-*/]?=\s*[-+]?\d+(?:\.\d+)?\b`,
    String.raw`\[[0-9.,\s+\-]+\]`,
    String.raw`\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\([^)]{0,80}\)`,
    String.raw`\b[a-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b`,
    String.raw`\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b`,
  ].join("|"), "g");

  function renderImplicitCode(html) {
    return html
      .split(/(<[^>]+>|\u0000CODE\d+\u0000)/g)
      .map((part) => {
        if (!part || part.startsWith("<") || /^\u0000CODE\d+\u0000$/.test(part)) return part;
        return part.replace(IMPLICIT_CODE_PATTERN, '<code class="skill-inline-code">$&</code>');
      })
      .join("");
  }

  function renderInline(value) {
    const codeTokens = [];
    let html = esc(value).replace(/`([^`]+)`/g, (_, code) => {
      const index = codeTokens.length;
      codeTokens.push(`<code class="skill-inline-code">${code}</code>`);
      return `\u0000CODE${index}\u0000`;
    });

    html = html
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    html = renderImplicitCode(html);

    return html.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => codeTokens[Number(index)] || "");
  }

  function isTableSeparator(line) {
    return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  }

  function isTableStart(lines, index) {
    return lines[index] && lines[index].includes("|") && isTableSeparator(lines[index + 1] || "");
  }

  function splitTableRow(line) {
    return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
  }

  function isListLine(line) {
    return /^\s*(?:[-*]\s+|\d+\.\s+)/.test(line);
  }

  function isBlockStart(lines, index) {
    const line = lines[index] || "";
    return /^```/.test(line.trim()) ||
      /^#{2,6}\s+/.test(line) ||
      /^>\s?/.test(line) ||
      /^\s*---+\s*$/.test(line) ||
      isListLine(line) ||
      isTableStart(lines, index);
  }

  function renderMarkdown(markdown) {
    const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      const trimmed = line.trim();
      if (!trimmed) {
        index += 1;
        continue;
      }

      if (/^```/.test(trimmed)) {
        index += 1;
        const code = [];
        while (index < lines.length && !/^```/.test(lines[index].trim())) {
          code.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        html.push(`<pre class="skill-code-block"><code>${esc(code.join("\n"))}</code></pre>`);
        continue;
      }

      const heading = line.match(/^(#{2,6})\s+(.+)$/);
      if (heading) {
        const level = Math.min(6, heading[1].length + 2);
        html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        index += 1;
        continue;
      }

      if (/^\s*---+\s*$/.test(line)) {
        html.push("<hr>");
        index += 1;
        continue;
      }

      if (isTableStart(lines, index)) {
        const header = splitTableRow(lines[index]);
        index += 2;
        const rows = [];
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
          rows.push(splitTableRow(lines[index]));
          index += 1;
        }
        html.push(`
          <div class="skill-table-wrap">
            <table class="skill-table">
              <thead><tr>${header.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead>
              <tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
            </table>
          </div>
        `);
        continue;
      }

      if (/^>\s?/.test(line)) {
        const quote = [];
        while (index < lines.length && /^>\s?/.test(lines[index])) {
          quote.push(lines[index].replace(/^>\s?/, "").trim());
          index += 1;
        }
        html.push(`<blockquote>${renderInline(quote.join(" "))}</blockquote>`);
        continue;
      }

      if (isListLine(line)) {
        const ordered = /^\s*\d+\.\s+/.test(line);
        const tag = ordered ? "ol" : "ul";
        const items = [];
        while (index < lines.length && isListLine(lines[index]) === true) {
          const itemLines = [lines[index].replace(/^\s*(?:[-*]\s+|\d+\.\s+)/, "").trim()];
          index += 1;
          while (
            index < lines.length &&
            lines[index].trim() &&
            !isListLine(lines[index]) &&
            !isBlockStart(lines, index)
          ) {
            itemLines.push(lines[index].trim());
            index += 1;
          }
          items.push(`<li>${renderInline(itemLines.join(" "))}</li>`);
        }
        html.push(`<${tag}>${items.join("")}</${tag}>`);
        continue;
      }

      const paragraph = [];
      while (index < lines.length && lines[index].trim() && !isBlockStart(lines, index)) {
        paragraph.push(lines[index].trim());
        index += 1;
      }
      html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    }

    return html.join("");
  }

  function domainClass(domain) {
    return "domain-" + norm(domain).replace(/\s+/g, "-");
  }

  function domainSort(a, b) {
    const ai = DOMAIN_ORDER.indexOf(a);
    const bi = DOMAIN_ORDER.indexOf(b);
    if (ai !== -1 || bi !== -1) return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    return a.localeCompare(b);
  }

  function skillMeta(skill) {
    return [skill.domain, skill.track].filter(Boolean).join(" / ");
  }

  function visibleSkills() {
    return skills.filter((skill) => {
      if (activeDomain !== "all" && skill.domain !== activeDomain) return false;
      if (!activeQuery) return true;
      const haystack = norm([
        skill.title,
        skill.domain,
        skill.track,
        skill.summary,
        CARD_SUMMARIES[skill.id],
        ...(skill.headings || []),
      ].join(" "));
      return haystack.includes(activeQuery);
    });
  }

  function renderStats() {
    const root = document.getElementById("skill-stats");
    if (!root) return;
    const items = [
      { label: "Skills", value: "90+" },
      { label: "Tasks", value: "150+" },
    ];
    root.innerHTML = items
      .map((item) => `<div class="skill-stat"><span>${item.value}</span><small>${item.label}</small></div>`)
      .join("");
  }

  function renderTabs() {
    const root = document.getElementById("skill-tabs");
    if (!root) return;
    const domains = ["all", ...new Set(skills.map((skill) => skill.domain))].sort((a, b) => {
      if (a === "all") return -1;
      if (b === "all") return 1;
      return domainSort(a, b);
    });
    root.innerHTML = domains
      .map((domain) => {
        const label = domain === "all" ? "All" : domain;
        const active = domain === activeDomain ? " active" : "";
        return `<button class="skill-tab${active}" type="button" data-domain="${esc(domain)}">${esc(label)}</button>`;
      })
      .join("");
    root.querySelectorAll(".skill-tab").forEach((button) => {
      button.addEventListener("click", () => {
        activeDomain = button.dataset.domain;
        activePage = 0;
        activeId = "";
        closeReader();
        renderAll();
      });
    });
  }

  function pageCountFor(items) {
    return Math.max(1, Math.ceil(items.length / skillPageSize()));
  }

  function renderList() {
    const root = document.getElementById("skill-list");
    if (!root) return;
    const shown = visibleSkills();

    if (!shown.length) {
      activeId = "";
      root.innerHTML = '<div class="skill-empty">No skills match the current filter.</div>';
      return;
    }

    const totalPages = pageCountFor(shown);
    activePage = Math.min(Math.max(activePage, 0), totalPages - 1);
    const ps = skillPageSize();
    const start = activePage * ps;
    const pageSkills = shown.slice(start, start + ps);
    if (!pageSkills.some((skill) => skill.id === activeId)) activeId = pageSkills[0] ? pageSkills[0].id : "";

    const cards = pageSkills
      .map((skill) => {
        const active = skill.id === activeId ? " active" : "";
        return `
          <button class="skill-card ${domainClass(skill.domain)}${active}" type="button" data-skill="${esc(skill.id)}">
            <span class="skill-card-meta">${esc(skillMeta(skill))}</span>
            <strong>${esc(skill.title)}</strong>
            <span class="skill-card-summary">${renderInline(CARD_SUMMARIES[skill.id] || skill.summary)}</span>
            <span class="skill-card-action" aria-hidden="true">Click to View</span>
          </button>
        `;
      })
      .join("");

    root.innerHTML = `
      <div class="skill-pagination" aria-label="Skill library pagination">
        <button class="skill-page-btn" type="button" data-page-action="prev" ${activePage === 0 ? "disabled" : ""}>Prev</button>
        <span class="skill-page-label">Page ${activePage + 1} of ${totalPages}</span>
        <button class="skill-page-btn" type="button" data-page-action="next" ${activePage === totalPages - 1 ? "disabled" : ""}>Next</button>
      </div>
      <div class="skill-page-items">
        ${cards}
      </div>
    `;

    root.querySelectorAll(".skill-card").forEach((button) => {
      button.addEventListener("click", () => {
        activeId = button.dataset.skill;
        renderAll();
        // On mobile the reader opens as a popup overlay (CSS); desktop ignores it.
        openReader();
      });
    });
    root.querySelectorAll("[data-page-action]").forEach((button) => {
      button.addEventListener("click", () => {
        activePage += button.dataset.pageAction === "next" ? 1 : -1;
        activeId = "";
        closeReader();
        renderAll();
      });
    });
  }

  function renderReader() {
    const root = document.getElementById("skill-reader");
    if (!root) return;
    const skill = skills.find((item) => item.id === activeId) || visibleSkills()[0];
    if (!skill) {
      root.innerHTML = "";
      return;
    }

    const fallback = (skill.snippets || [])
      .map((snippet) => `
        <section class="skill-snippet">
          <h4>${renderInline(snippet.label)}</h4>
          <p>${renderInline(snippet.text)}</p>
        </section>
      `)
      .join("");
    const body = skill.markdown
      ? `<div class="skill-markdown">${renderMarkdown(skill.markdown)}</div>`
      : `<div class="skill-snippets">${fallback}</div>`;
    root.className = `skill-reader ${domainClass(skill.domain)}`;
    root.innerHTML = `
      <button class="skill-reader-close" type="button" aria-label="Close">&times;</button>
      <header class="skill-reader-head">
        <div class="skill-card-meta">${esc(skillMeta(skill))}</div>
        <h3>${esc(skill.title)}</h3>
        <p>${renderInline(skill.summary)}</p>
      </header>
      ${body}
    `;
    root.querySelector(".skill-reader-close").addEventListener("click", closeReader);
  }

  // Mobile-only popup state: clicking a skill card adds this class (CSS turns the
  // reader into a full-screen overlay); the X / Esc / tab / search / page changes
  // remove it. On desktop the reader is always inline, so the class is a no-op.
  function openReader() { document.body.classList.add("skill-reader-open"); }
  function closeReader() { document.body.classList.remove("skill-reader-open"); }

  function renderAll() {
    renderTabs();
    renderList();
    renderReader();
  }

  async function init() {
    const list = document.getElementById("skill-list");
    const search = document.getElementById("skill-search");
    if (!list || !search) return;

    try {
      const response = await fetch(DATA_URL);
      if (!response.ok) throw new Error("skill data unavailable");
      const data = await response.json();
      skills = (data.skills || []).sort((a, b) =>
        domainSort(a.domain, b.domain) || a.track.localeCompare(b.track) || a.title.localeCompare(b.title)
      );
      activeId = skills[0] ? skills[0].id : "";
      renderStats();
      renderAll();
    } catch (_) {
      list.innerHTML = '<div class="skill-empty">Skill library data is unavailable.</div>';
    }

    search.addEventListener("input", () => {
      activeQuery = norm(search.value);
      activePage = 0;
      activeId = "";
      closeReader();
      renderAll();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeReader();
    });

    // Re-paginate when crossing the mobile breakpoint (e.g., rotating the phone)
    let lastMobile = isMobile();
    window.addEventListener("resize", () => {
      const m = isMobile();
      if (m === lastMobile) return;
      lastMobile = m;
      activePage = 0;
      closeReader();
      renderAll();
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
