/**
 * Shared syntax helpers for lightweight Python code previews.
 */
(function () {
  "use strict";

  const KEYWORDS = new Set([
    "from", "import", "def", "return", "for", "in", "if", "elif", "else", "while",
    "assert", "None", "True", "False", "and", "or", "not", "is", "with", "as",
    "lambda", "raise", "continue", "break", "pass", "class", "print", "try",
    "except", "finally", "yield", "global", "del",
  ]);

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function tokenizePythonLine(line, options = {}) {
    const prefix = options.prefix || "t";
    let html = "";
    let i = 0;
    const isIdent = (char) => /[A-Za-z0-9_]/.test(char);

    while (i < line.length) {
      const char = line[i];
      if (char === "#") {
        html += `<span class="${prefix}-c">${escapeHtml(line.slice(i))}</span>`;
        break;
      }
      if (char === '"' || char === "'") {
        let j = i + 1;
        while (j < line.length && line[j] !== char) {
          if (line[j] === "\\") j += 1;
          j += 1;
        }
        j = Math.min(j + 1, line.length);
        html += `<span class="${prefix}-s">${escapeHtml(line.slice(i, j))}</span>`;
        i = j;
        continue;
      }
      if (/[0-9]/.test(char) && !(i > 0 && isIdent(line[i - 1]))) {
        let j = i;
        while (j < line.length && /[0-9._]/.test(line[j])) j += 1;
        html += `<span class="${prefix}-n">${escapeHtml(line.slice(i, j))}</span>`;
        i = j;
        continue;
      }
      if (/[A-Za-z_]/.test(char)) {
        let j = i;
        while (j < line.length && isIdent(line[j])) j += 1;
        const word = line.slice(i, j);
        if (KEYWORDS.has(word)) html += `<span class="${prefix}-k">${word}</span>`;
        else if (line[j] === "(") html += `<span class="${prefix}-f">${escapeHtml(word)}</span>`;
        else html += escapeHtml(word);
        i = j;
        continue;
      }
      html += escapeHtml(char);
      i += 1;
    }

    return html || "&#8203;";
  }

  window.AspireSyntax = { escapeHtml, tokenizePythonLine };
})();
