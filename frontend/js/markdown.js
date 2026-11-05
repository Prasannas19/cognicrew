/* CogniCrew Comprehensive Markdown & CodeBlock Renderer.
 * Robust, safe, and aesthetic markdown rendering:
 * - Full fenced code block support (```lang ... ```)
 * - Dedicated CodeBlock component with dark panel, language badge, and isolated Copy button
 * - Preserved whitespace, tabs, and indentation
 * - Language auto-detection fallback (Python, JS, TS, Java, C, C++, HTML, CSS, SQL, Bash, JSON)
 * - Mathematical formatting support with safe DOM generation (NO eval / NO exec)
 * - Tables, blockquotes, headings, lists, inline code, bold, italic
 * - Strict XSS safety: DOM node construction, createTextNode for all content.
 */
(function () {
  'use strict';

  const STYLE_ID = 'cc-md-styles';
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = [
      '.cc-md { font-size: 14.5px; line-height: 1.7; color: var(--foreground); }',
      '.cc-md p { margin: 0.5rem 0; }',
      '.cc-md h1, .cc-md h2, .cc-md h3, .cc-md h4 { margin: 1rem 0 0.4rem; font-weight: 600; color: var(--primary); font-family: var(--font-display, Georgia, serif); }',
      '.cc-md h1 { font-size: 1.35rem; }',
      '.cc-md h2 { font-size: 1.15rem; border-bottom: 1px solid var(--line); padding-bottom: 0.3rem; margin-top: 1.25rem; }',
      '.cc-md h3 { font-size: 1.02rem; }',
      '.cc-md h4 { font-size: 0.95rem; }',
      '.cc-md table { border-collapse: collapse; margin: 0.8rem 0; font-size: 13.5px; width: 100%; max-width: 100%; display: block; overflow-x: auto; border-radius: 8px; border: 1px solid var(--line); }',
      '.cc-md th, .cc-md td { border: 1px solid var(--line); padding: 8px 14px; text-align: left; vertical-align: top; }',
      '.cc-md th { background: var(--secondary); font-weight: 600; color: var(--foreground); }',
      '.cc-md tr:nth-child(even) td { background: rgba(0, 0, 0, 0.015); }',
      '.cc-md ul, .cc-md ol { margin: 0.5rem 0 0.5rem 1.4rem; padding: 0; }',
      '.cc-md li { margin: 0.25rem 0; }',
      '.cc-md blockquote { border-left: 3px solid var(--primary); background: rgba(0, 143, 134, 0.05); padding: 0.6rem 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0; color: var(--ink-soft); font-style: italic; }',
      '.cc-md code.cc-inline-code { font-family: var(--font-mono, monospace); font-size: 0.88em; background: var(--secondary); color: var(--accent-teal-dark, #006f69); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--line); }',
      '.cc-md strong { font-weight: 600; color: var(--foreground); }',
      '.cc-md em { font-style: italic; }',
      '.cc-code-panel { margin: 1rem 0; border-radius: 0.75rem; background: #0f172a; border: 1px solid #334155; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25); }',
      '.cc-code-header { display: flex; justify-content: space-between; align-items: center; background: #1e293b; padding: 8px 16px; border-bottom: 1px solid #334155; }',
      '.cc-code-header-left { display: flex; align-items: center; gap: 8px; }',
      '.cc-code-badge { font-family: var(--font-mono, monospace); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #38bdf8; background: rgba(56, 189, 248, 0.12); padding: 2px 8px; border-radius: 4px; }',
      '.cc-code-filename { font-family: var(--font-mono, monospace); font-size: 11px; color: #94a3b8; }',
      '.cc-code-copy-btn { font-family: var(--font-sans, sans-serif); font-size: 12px; font-weight: 500; color: #cbd5e1; background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 6px; padding: 4px 10px; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; transition: all 0.15s ease; }',
      '.cc-code-copy-btn:hover { background: rgba(255, 255, 255, 0.16); color: #ffffff; }',
      '.cc-code-copy-btn.copied { background: #059669; border-color: #059669; color: #ffffff; }',
      '.cc-code-pre { margin: 0; padding: 14px 18px; background: #0f172a; color: #f8fafc; font-family: var(--font-mono, monospace); font-size: 13.5px; line-height: 1.6; overflow-x: auto; white-space: pre; tab-size: 4; -webkit-overflow-scrolling: touch; }',
      '.cc-code-inner { font-family: inherit; font-size: inherit; white-space: pre; color: inherit; }',
      '.cc-token-keyword { color: #c678dd; font-weight: 600; }',
      '.cc-token-type { color: #4ec9b0; }',
      '.cc-token-builtin { color: #e5c07b; }',
      '.cc-token-string { color: #98c379; }',
      '.cc-token-number { color: #d19a66; }',
      '.cc-token-comment { color: #64748b; font-style: italic; }',
      '.cc-token-fn { color: #61afef; }',
      '.cc-token-tag { color: #e06c75; font-weight: 500; }',
      '.cc-token-attr { color: #d19a66; }',
      '.cc-token-prop { color: #e5c07b; }',
      '.cc-token-op { color: #56b6c2; }',
      '.cc-token-punct { color: #94a3b8; }',
      '.cc-math-panel { background: rgba(0, 143, 134, 0.04); border-left: 3px solid var(--primary); padding: 0.75rem 1.1rem; border-radius: 0 0.75rem 0.75rem 0; margin: 0.8rem 0; font-family: var(--font-mono, monospace); font-size: 13.5px; line-height: 1.65; overflow-x: auto; color: var(--ink); }',
      '.cc-step-badge { display: inline-block; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; padding: 2px 8px; border-radius: 4px; background: var(--secondary); color: var(--primary); margin-right: 6px; }'
    ].join('\n');
    document.head.appendChild(style);
  }

  function detectLanguage(code, hint) {
    if (hint && hint.trim()) {
      const h = hint.trim().toLowerCase();
      if (h === 'py' || h === 'python') return 'Python';
      if (h === 'js' || h === 'javascript') return 'JavaScript';
      if (h === 'ts' || h === 'typescript') return 'TypeScript';
      if (h === 'java') return 'Java';
      if (h === 'c') return 'C';
      if (h === 'cpp' || h === 'c++') return 'C++';
      if (h === 'html') return 'HTML';
      if (h === 'css') return 'CSS';
      if (h === 'sql') return 'SQL';
      if (h === 'sh' || h === 'bash' || h === 'shell') return 'Bash';
      if (h === 'json') return 'JSON';
      if (h === 'xml') return 'XML';
      return hint.toUpperCase();
    }
    // Heuristic detection from code content
    if (/(^import |^from \w+ import |def \w+\(|plt\.plot|np\.\w+|print\(|__name__)/m.test(code)) return 'Python';
    if (/(public class |System\.out\.println|public static void main)/.test(code)) return 'Java';
    if (/(const \w+ =|let \w+ =|function \w+\(|console\.log\(|=>)/.test(code)) return 'JavaScript';
    if (/(SELECT\s+.+\s+FROM\s+|INSERT INTO|CREATE TABLE)/i.test(code)) return 'SQL';
    if (/(^<!DOCTYPE html>|<html|<div|<script)/i.test(code)) return 'HTML';
    if (/(\{|\})/.test(code) && /"[\w_-]+":\s*["\d\[\{]/.test(code)) return 'JSON';
    if (/(#!\/bin\/(bash|sh)|curl\s+|sudo\s+|chmod\s+)/.test(code)) return 'Bash';
    return 'Code';
  }

  function highlightCode(targetEl, codeText, langName) {
    const lang = (langName || '').toLowerCase();
    targetEl.innerHTML = '';
    const frag = document.createDocumentFragment();

    const appendToken = (type, text) => {
      if (!text) return;
      if (type && type !== 'plain') {
        const span = document.createElement('span');
        span.className = 'cc-token-' + type;
        span.textContent = text;
        frag.appendChild(span);
      } else {
        frag.appendChild(document.createTextNode(text));
      }
    };

    if (lang === 'json') {
      const rx = /("(?:\\.|[^"\\])*"(?:\s*:)?)|(\b-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\b(?:true|false|null)\b)|([{}[\],:])|([^"0-9a-zA-Z{}[\],:\s]+|\s+)/g;
      let m;
      while ((m = rx.exec(codeText)) !== null) {
        if (m[1]) {
          if (m[1].trim().endsWith(':')) appendToken('prop', m[1]);
          else appendToken('string', m[1]);
        } else if (m[2]) appendToken('number', m[2]);
        else if (m[3]) appendToken('keyword', m[3]);
        else if (m[4]) appendToken('punct', m[4]);
        else appendToken('plain', m[0]);
      }
      targetEl.appendChild(frag);
      return;
    }

    if (lang === 'html' || lang === 'xml') {
      const rx = /(<!--[\s\S]*?-->)|(<\/?[a-zA-Z0-9\-]+)|(\s+[a-zA-Z0-9_\-]+(?==))|("(?:\\.|[^"\\])*"|'[^']*')|([/>=])|([^<>"'/=\s]+|\s+)/g;
      let m;
      while ((m = rx.exec(codeText)) !== null) {
        if (m[1]) appendToken('comment', m[1]);
        else if (m[2]) appendToken('tag', m[2]);
        else if (m[3]) appendToken('attr', m[3]);
        else if (m[4]) appendToken('string', m[4]);
        else if (m[5]) appendToken('punct', m[5]);
        else appendToken('plain', m[0]);
      }
      targetEl.appendChild(frag);
      return;
    }

    if (lang === 'css') {
      const rx = /(\/\*[\s\S]*?\*\/)|([.#]?[a-zA-Z0-9_\-]+(?=\s*\{))|([a-zA-Z\-]+(?=\s*:))|("(?:\\.|[^"\\])*"|'[^']*')|(#[0-9a-fA-F]{3,8}\b)|(-?\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms)?)|([{}:;,])|([^a-zA-Z0-9\s{}():;,"'#]+|\s+)/g;
      let m;
      while ((m = rx.exec(codeText)) !== null) {
        if (m[1]) appendToken('comment', m[1]);
        else if (m[2]) appendToken('tag', m[2]);
        else if (m[3]) appendToken('prop', m[3]);
        else if (m[4]) appendToken('string', m[4]);
        else if (m[5] || m[6]) appendToken('number', m[5] || m[6]);
        else if (m[7]) appendToken('punct', m[7]);
        else appendToken('plain', m[0]);
      }
      targetEl.appendChild(frag);
      return;
    }

    const isPy = (lang === 'python' || lang === 'py');
    const isBash = (lang === 'bash' || lang === 'sh' || lang === 'shell');
    const isSql = (lang === 'sql');
    const isJava = (lang === 'java');
    const isJs = (lang === 'javascript' || lang === 'js' || lang === 'typescript' || lang === 'ts');
    const isC = (lang === 'c' || lang === 'c++' || lang === 'cpp');

    const pyKeywords = new Set(['def','class','import','from','return','if','elif','else','for','while','try','except','finally','with','as','lambda','yield','break','continue','pass','raise','assert','async','await','global','nonlocal','in','is','not','and','or']);
    const pyBuiltins = new Set(['True','False','None','self','cls','int','float','str','list','dict','set','tuple','bool','print','len','range','enumerate','zip','open','isinstance','type','super']);

    const jsKeywords = new Set(['const','let','var','function','return','if','else','for','while','do','switch','case','default','break','continue','try','catch','finally','throw','new','class','extends','import','export','from','default','async','await','typeof','instanceof','in','of','yield','interface','type','implements','private','public','protected','readonly','static']);
    const jsBuiltins = new Set(['true','false','null','undefined','NaN','this','window','document','console','Math','JSON','Promise','Array','Object','String','Number','Boolean','Set','Map']);

    const javaKeywords = new Set(['public','private','protected','static','final','void','class','interface','enum','extends','implements','new','return','if','else','for','while','do','switch','case','default','break','continue','try','catch','finally','throw','throws','package','import','this','super','synchronized','volatile','transient','native','abstract']);
    const javaTypes = new Set(['int','long','short','byte','float','double','boolean','char','String','Integer','Long','Boolean','List','Map','Set','ArrayList','HashMap','System','Object','Scanner','Exception','Thread']);

    const cKeywords = new Set(['int','char','float','double','void','long','short','signed','unsigned','struct','union','enum','typedef','const','static','extern','auto','register','volatile','sizeof','if','else','switch','case','default','for','while','do','break','continue','return','goto','namespace','using','template','typename','virtual','override','public','private','protected','class','bool','true','false']);

    const sqlKeywords = new Set(['select','from','where','and','or','insert','into','values','update','set','delete','join','left','right','inner','outer','group','by','order','asc','desc','limit','offset','having','create','table','drop','alter','primary','key','foreign','references','not','null','distinct','as','like','in','is','between','union','all','count','sum','avg','min','max','view','index']);

    const bashKeywords = new Set(['if','then','else','elif','fi','for','in','do','done','while','until','case','esac','function','return','exit','echo','cd','ls','cat','grep','curl','wget','sudo','chmod','chown','mkdir','rm','cp','mv','touch','export','source','set','unset']);

    const masterRx = /(\/\*[\s\S]*?\*\/|\/\/[^\n]*|#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'[^']*'|`[\s\S]*?`)|(@[a-zA-Z_]\w*|#[a-zA-Z_]\w*)|(\$[a-zA-Z0-9_]+|\$\{[^}]+\})|([a-zA-Z_]\w*)|(\b\d+(?:\.\d+)?[fFlLuU]?\b)|([+\-*/%=&|!<>^~?:;,.(){}\[\]]+)|(\s+)/g;

    let m;
    while ((m = masterRx.exec(codeText)) !== null) {
      if (m[1]) {
        appendToken('comment', m[1]);
      } else if (m[2]) {
        appendToken('string', m[2]);
      } else if (m[3]) {
        appendToken('prop', m[3]);
      } else if (m[4]) {
        appendToken('prop', m[4]);
      } else if (m[5]) {
        const w = m[5];
        const lw = w.toLowerCase();
        if (isPy) {
          if (pyKeywords.has(w)) appendToken('keyword', w);
          else if (pyBuiltins.has(w)) appendToken('builtin', w);
          else appendToken('plain', w);
        } else if (isJava) {
          if (javaKeywords.has(w)) appendToken('keyword', w);
          else if (javaTypes.has(w)) appendToken('type', w);
          else appendToken('plain', w);
        } else if (isJs) {
          if (jsKeywords.has(w)) appendToken('keyword', w);
          else if (jsBuiltins.has(w)) appendToken('builtin', w);
          else appendToken('plain', w);
        } else if (isC) {
          if (cKeywords.has(w)) appendToken('keyword', w);
          else appendToken('plain', w);
        } else if (isSql) {
          if (sqlKeywords.has(lw)) appendToken('keyword', w);
          else appendToken('plain', w);
        } else if (isBash) {
          if (bashKeywords.has(lw)) appendToken('keyword', w);
          else appendToken('plain', w);
        } else {
          appendToken('plain', w);
        }
      } else if (m[6]) {
        appendToken('number', m[6]);
      } else if (m[7]) {
        appendToken('punct', m[7]);
      } else {
        appendToken('plain', m[0]);
      }
    }
    targetEl.appendChild(frag);
  }

  function createCodeBlock(codeText, langHint, filename) {
    const langName = detectLanguage(codeText, langHint);

    const panel = document.createElement('div');
    panel.className = 'cc-code-panel';

    const header = document.createElement('div');
    header.className = 'cc-code-header';

    const left = document.createElement('div');
    left.className = 'cc-code-header-left';

    const badge = document.createElement('span');
    badge.className = 'cc-code-badge';
    badge.textContent = langName;
    left.appendChild(badge);

    if (filename) {
      const fn = document.createElement('span');
      fn.className = 'cc-code-filename';
      fn.textContent = filename;
      left.appendChild(fn);
    }
    header.appendChild(left);

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'cc-code-copy-btn';
    copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg><span>Copy</span>';

    copyBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      const textToCopy = codeText;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(textToCopy).then(onCopied).catch(fallbackCopy);
      } else {
        fallbackCopy();
      }

      function onCopied() {
        copyBtn.classList.add('copied');
        copyBtn.innerHTML = '<span>Copied ✓</span>';
        setTimeout(function () {
          copyBtn.classList.remove('copied');
          copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg><span>Copy</span>';
        }, 2000);
      }

      function fallbackCopy() {
        const ta = document.createElement('textarea');
        ta.value = textToCopy;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        try {
          document.execCommand('copy');
          onCopied();
        } catch (_) {}
        document.body.removeChild(ta);
      }
    });

    header.appendChild(copyBtn);
    panel.appendChild(header);

    const pre = document.createElement('pre');
    pre.className = 'cc-code-pre';
    const code = document.createElement('code');
    code.className = 'cc-code-inner';
    highlightCode(code, codeText, langName);
    pre.appendChild(code);
    panel.appendChild(pre);

    return panel;
  }

  function isTableSeparator(line) {
    return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(line);
  }

  function isTableRow(line) {
    const t = line.trim();
    return t.startsWith('|') && t.endsWith('|') && !isTableSeparator(t);
  }

  function splitRow(line) {
    return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
  }

  function renderInline(parent, text) {
    // Split on **bold**, *italic*, `inline code`
    const rx = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g;
    let last = 0, m;
    while ((m = rx.exec(text)) !== null) {
      if (m.index > last) {
        parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      }
      const tok = m[0];
      if (tok.startsWith('**')) {
        const el = document.createElement('strong');
        el.textContent = tok.slice(2, -2);
        parent.appendChild(el);
      } else if (tok.startsWith('`')) {
        const el = document.createElement('code');
        el.className = 'cc-inline-code';
        el.textContent = tok.slice(1, -1);
        parent.appendChild(el);
      } else {
        const el = document.createElement('em');
        el.textContent = tok.slice(1, -1);
        parent.appendChild(el);
      }
      last = m.index + tok.length;
    }
    if (last < text.length) {
      parent.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  function renderMarkdown(md) {
    const root = document.createElement('div');
    root.className = 'cc-md';
    if (!md) return root;

    const lines = String(md).split(/\r?\n/);
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // 1. Fenced Code Blocks (```lang ... ```)
      const fenceMatch = line.match(/^\s*```([a-zA-Z0-9_-]*)\s*$/);
      if (fenceMatch) {
        const langHint = fenceMatch[1];
        const codeLines = [];
        i += 1;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
          codeLines.push(lines[i]);
          i += 1;
        }
        if (i < lines.length) i += 1; // skip closing fence
        const codeText = codeLines.join('\n');
        root.appendChild(createCodeBlock(codeText, langHint));
        continue;
      }

      // 2. Math display block ($$ ... $$)
      if (/^\s*\$\$\s*$/.test(line)) {
        const mathLines = [];
        i += 1;
        while (i < lines.length && !/^\s*\$\$\s*$/.test(lines[i])) {
          mathLines.push(lines[i]);
          i += 1;
        }
        if (i < lines.length) i += 1;
        const panel = document.createElement('div');
        panel.className = 'cc-math-panel';
        panel.textContent = mathLines.join('\n');
        root.appendChild(panel);
        continue;
      }

      // 3. Markdown Tables
      if (isTableRow(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const headTr = document.createElement('tr');
        for (const cell of splitRow(line)) {
          const th = document.createElement('th');
          renderInline(th, cell);
          headTr.appendChild(th);
        }
        thead.appendChild(headTr);
        table.appendChild(thead);
        const tbody = document.createElement('tbody');
        i += 2;
        while (i < lines.length && isTableRow(lines[i])) {
          const tr = document.createElement('tr');
          for (const cell of splitRow(lines[i])) {
            const td = document.createElement('td');
            renderInline(td, cell);
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
          i += 1;
        }
        table.appendChild(tbody);
        root.appendChild(table);
        continue;
      }

      // 4. Headings
      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        const hLevel = heading[1].length;
        const h = document.createElement('h' + hLevel);
        const titleText = heading[2].trim();

        // Academic math step labels badge styling (e.g. Given, Formula, Substitution, Therefore)
        const stepMatch = titleText.match(/^(Given|Formula|Substitution|Calculation|Simplification|Therefore|Problem from document|Derived solution):?(.*)$/i);
        if (stepMatch) {
          const badge = document.createElement('span');
          badge.className = 'cc-step-badge';
          badge.textContent = stepMatch[1];
          h.appendChild(badge);
          if (stepMatch[2] && stepMatch[2].trim()) {
            renderInline(h, ' ' + stepMatch[2].trim());
          }
        } else {
          renderInline(h, titleText);
        }
        root.appendChild(h);
        i += 1;
        continue;
      }

      // 5. Blockquotes
      if (/^\s*>\s*/.test(line)) {
        const quote = document.createElement('blockquote');
        const quoteLines = [];
        while (i < lines.length && /^\s*>\s*/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^\s*>\s?/, ''));
          i += 1;
        }
        renderInline(quote, quoteLines.join(' '));
        root.appendChild(quote);
        continue;
      }

      // 6. Bullet lists
      if (/^\s*[-*]\s+/.test(line)) {
        const ul = document.createElement('ul');
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
          const li = document.createElement('li');
          renderInline(li, lines[i].replace(/^\s*[-*]\s+/, ''));
          ul.appendChild(li);
          i += 1;
        }
        root.appendChild(ul);
        continue;
      }

      // 7. Numbered lists
      if (/^\s*\d+[.)]\s+/.test(line)) {
        const ol = document.createElement('ol');
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          const li = document.createElement('li');
          renderInline(li, lines[i].replace(/^\s*\d+[.)]\s+/, ''));
          ol.appendChild(li);
          i += 1;
        }
        root.appendChild(ol);
        continue;
      }

      // 8. Empty lines
      if (line.trim() === '') {
        i += 1;
        continue;
      }

      // 9. Paragraph: merge consecutive plain text lines
      const p = document.createElement('p');
      let buf = line;
      while (
        i + 1 < lines.length &&
        lines[i + 1].trim() !== '' &&
        !/^\s*```/.test(lines[i + 1]) &&
        !/^\s*\$\$/.test(lines[i + 1]) &&
        !isTableRow(lines[i + 1]) &&
        !/^(#{1,4})\s+/.test(lines[i + 1]) &&
        !/^\s*[-*]\s+/.test(lines[i + 1]) &&
        !/^\s*\d+[.)]\s+/.test(lines[i + 1]) &&
        !/^\s*>\s*/.test(lines[i + 1])
      ) {
        buf += ' ' + lines[i + 1].trim();
        i += 1;
      }
      renderInline(p, buf);
      root.appendChild(p);
      i += 1;
    }

    return root;
  }

  window.renderMarkdown = renderMarkdown;
  window.createCodeBlock = createCodeBlock;
})();
