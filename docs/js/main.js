const themes = {
  "Classic Bone": {
    "--bg-color": "#F2EDE4",
    "--card-bg-color": "#F7F2EA",
    "--text-color": "#595149",
    "--title-color": "#3A2E25",
    "--secondary-text": "#866c5b",
    "--nav-color": "#3A2E25",
    "--nav-text-color": "#F2EDE4",
    "--accent-color": "#b65243",
    "--input-bg-color": "#fdf5e8",
    "--btn-hover-color": "#a04835",
    "--success-color": "#28a745",
    "--footer-overlay": "rgb(58 46 37 / 51%)",
    "img": "./imgs/home-classic-theme.png"
  },
  "Dark reading mode": {
    "--bg-color": "#3D2535",
    "--card-bg-color": "#2A1A20",
    "--text-color": "#e6e6e9",
    "--title-color": "#f2f4f5",
    "--secondary-text": "#a89bb0",
    "--nav-color": "#1A1015",
    "--nav-text-color": "#f2f4f5",
    "--accent-color": "#b0bec5",
    "--input-bg-color": "#2b1a25",
    "--btn-hover-color": "#493a49",
    "--success-color": "#64b5f6",
    "--footer-overlay": "rgba(26, 16, 21, 0.55)",

    "img": "./imgs/home-dark-theme.png"
  },
  "Soft pastel": {
    "--bg-color": "#F5EFE6",
    "--card-bg-color": "#fff7ec",
    "--text-color": "#394b64",
    "--title-color": "rgb(21, 19, 37)",
    "--nav-color": "#394b64",
    "--secondary-text": "#6b7f99",
    "--nav-text-color": "#F5EFE6",
    "--accent-color": "#79acd0",
    "--input-bg-color": "#afc4d25e",
    "--btn-hover-color": "#5a8ba8",
    "--success-color": "#81c784",
    "--footer-overlay": "rgba(57, 75, 100, 0.35)",

    "img": "./imgs/home-pastel-theme.png"
  }
};

function changeTheme(themeName) {
  const theme = themes[themeName];

  for (let varName in theme) {
    if (varName.startsWith("--")) {
      document.documentElement.style.setProperty(varName, theme[varName]);
    }
  }

  const img = document.getElementById("theme-img");
  if (img) img.src = theme.img;

  document.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
  document.querySelector(`[data-theme="${themeName}"]`).classList.add("active");
  localStorage.setItem("theme", themeName);
}

document.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("theme") || "Classic Bone";
  changeTheme(saved);

  document.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => changeTheme(btn.dataset.theme));
  });
});

//show of all sections
const reveals = document.querySelectorAll(".reveal");

window.addEventListener("scroll", () => {
  reveals.forEach((el) => {
    const windowHeight = window.innerHeight;
    const elementTop = el.getBoundingClientRect().top;
    const visiblePoint = 100;

    if (elementTop < windowHeight - visiblePoint) {
      el.classList.add("active");
    }
  });
});
//navbar
const navbar = document.querySelector('.navbar');
const aboutSection = document.getElementById('About');

window.addEventListener('scroll', () => {
  const aboutBottom = aboutSection.getBoundingClientRect().bottom;

  if (aboutBottom > 0) {
    navbar.style.opacity = '1';
    navbar.style.pointerEvents = 'auto';
  } else {
    navbar.style.opacity = '0';
    navbar.style.pointerEvents = 'none';
  }
});

navbar.style.transition = 'opacity 0.3s ease';


// ── Font Size Control ──//
const fontSizes = [16, 18, 20, 22];
let currentFontIndex = 1;

const savedFontIndex = localStorage.getItem('fontIndex');
if (savedFontIndex) {
  currentFontIndex = parseInt(savedFontIndex);
  document.body.style.fontSize = fontSizes[currentFontIndex] + 'px';
}

document.getElementById('fontIncrease').addEventListener('click', () => {
  if (currentFontIndex < fontSizes.length - 1) {
    currentFontIndex++;
    document.body.style.fontSize = fontSizes[currentFontIndex] + 'px';
    localStorage.setItem('fontIndex', currentFontIndex);
  }
});

document.getElementById('fontDecrease').addEventListener('click', () => {
  if (currentFontIndex > 0) {
    currentFontIndex--;
    document.body.style.fontSize = fontSizes[currentFontIndex] + 'px';
    localStorage.setItem('fontIndex', currentFontIndex);
  }
});
// ── Hide Font Controls Near Footer ──//
const fontControls = document.getElementById('fontControls');
const footer = document.querySelector('footer');

const observerOptions = {
  root: null,
  threshold: 0.1
};

const footerObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      fontControls.style.opacity = '0';
      fontControls.style.pointerEvents = 'none';
    } else {
      fontControls.style.opacity = '1';
      fontControls.style.pointerEvents = 'auto';
    }
  });
}, observerOptions);

if (footer) {
  footerObserver.observe(footer);
}
// ── Step Cards Animation ──//
const stepCards = document.querySelectorAll('.step-card');

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.2 });

stepCards.forEach((card, index) => {
  card.style.transitionDelay = `${index * 0.20}s`;
  observer.observe(card);
});
// ── Tips Animation ──//

const tipCards = document.querySelectorAll('.tip-card');

tipCards.forEach((card, index) => {
  card.style.transitionDelay = `${index * 0.15}s`;
  observer.observe(card);
});
// API Configuration
const API_BASE_URL = "http://127.0.0.1:8000";

let lastProcessResult = null;

// API Functions
async function callAPI(endpoint, text) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}

async function simplifyText(text) { return await callAPI("/simplify", text); }
async function processText(text) { return await callAPI("/process", text); }

// Main Send Functions
document.getElementById("submitBtn").addEventListener("click", () => sendTextToAI("process"));
document.getElementById("analyzeBtn").addEventListener("click", () => sendTextToAI("analyze"));
document.getElementById("simplifyBtn").addEventListener("click", () => sendTextToAI("simplify"));

async function sendTextToAI(actionType) {
  const text = document.getElementById("textInput").value;
  if (!text.trim()) { alert("Please enter some text"); return; }

  document.getElementById("loadingSpinner").style.display = "block";

  try {
    let data;

    if (actionType === "process") {
      data = await processText(text);
      lastProcessResult = data;
    } else if (actionType === "analyze") {
      // No API call — reuse what /process already returned
      if (!lastProcessResult) { alert("Please process the text first."); return; }
      data = lastProcessResult;
    } else if (actionType === "simplify") {
      data = await simplifyText(text);
    }

    document.getElementById("loadingSpinner").style.display = "none";
    showResult(data, actionType);
    document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth" });

  } catch (error) {
    document.getElementById("loadingSpinner").style.display = "none";
    alert("Error processing text. Make sure the API server is running.");
    console.error(error);
  }
}

// Replace the old highlightWords function with this:
function highlightWords(text, wordObjects, cssClass) {
  if (!wordObjects || wordObjects.length === 0) return text;

  // Build a lookup by word string for metadata access
  const metaMap = {};
  wordObjects.forEach(obj => {
    const key = (typeof obj === "string" ? obj : obj.word).toLowerCase();
    metaMap[key] = obj;
  });

  const words = Object.keys(metaMap);
  if (words.length === 0) return text;

  // Sort longest first to avoid partial matches
  words.sort((a, b) => b.length - a.length);
  const pattern = words.map(w => `\\b${w}\\b`).join("|");
  const regex = new RegExp(pattern, "gi");

  return text.replace(regex, match => {
    const meta = metaMap[match.toLowerCase()];

    if (typeof meta === "string" || !meta.syllables) {
      // survived words — simple highlight, no tooltip
      return `<span class="${cssClass}">${match}</span>`;
    }

    // original hard words — full tooltip + click-to-simplify
    const syllables = meta.syllables || match;
    const definition = meta.definition || "";
    const level = meta.difficulty_level || "";
    const reasons = (meta.reasons && meta.reasons.length > 0)
      ? meta.reasons.join(" · ")
      : "";

    return `<span 
      class="${cssClass} hard-word-interactive" 
      data-word="${match}"
      data-syllables="${syllables}"
      data-definition="${definition.replace(/"/g, '&quot;')}"
      data-level="${level}"
      data-reasons="${reasons.replace(/"/g, '&quot;')}"
      title=""
    >${match}</span>`;
  });
}

function initInteractiveWords(containerEl) {
  let tooltip = document.getElementById("word-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "word-tooltip";
    tooltip.style.cssText = `
      position: fixed;
      z-index: 9999;
      background: var(--card-bg-color);
      border: 1px solid var(--accent-color);
      border-radius: 10px;
      padding: 10px 14px;
      max-width: 240px;
      font-size: 0.85rem;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.15s ease;
      color: var(--text-color);
    `;
    document.body.appendChild(tooltip);
  }

  containerEl.querySelectorAll(".hard-word-interactive").forEach(el => {
    const syllables = el.dataset.syllables;
    const definition = el.dataset.definition;
    const reasons = el.dataset.reasons;

    el.addEventListener("mouseenter", async (e) => {
      if (!el.dataset.definitionLoaded) {
        try {
          const res = await fetch(`${API_BASE_URL}/define`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: el.dataset.word }),
          });
          const data = await res.json();
          el.dataset.definition = data.success ? data.definition : "No definition available.";
          el.dataset.definitionLoaded = "true";
        } catch (_) {
          el.dataset.definition = "No definition available.";
          el.dataset.definitionLoaded = "true";
        }
      }

      const definition = el.dataset.definition || "";
      tooltip.innerHTML = `
        <div style="font-size:1rem; font-weight:700; letter-spacing:0.05em;
                    margin-bottom:4px; color:var(--accent-color)">
          ${el.dataset.syllables}
        </div>
        ${definition
          ? `<div style="margin-bottom:6px; line-height:1.4">${definition}</div>`
          : `<div style="opacity:0.5; font-style:italic">No definition available.</div>`}
        ${el.dataset.reasons
          ? `<div style="font-size:0.75rem; opacity:0.7">${el.dataset.reasons}</div>`
          : ""}
    `;
      tooltip.style.opacity = "1";
      positionTooltip(e, tooltip);
    });

    el.addEventListener("mousemove", (e) => positionTooltip(e, tooltip));
    el.addEventListener("mouseleave", () => { tooltip.style.opacity = "0"; });
  });
}

function positionTooltip(e, tooltip) {
  const offset = 14;
  let x = e.clientX + offset;
  let y = e.clientY + offset;

  // Keep tooltip inside viewport
  if (x + 250 > window.innerWidth) x = e.clientX - 250 - offset;
  if (y + 120 > window.innerHeight) y = e.clientY - 120 - offset;

  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";
}

function showResult(data, actionType) {
  const resultsSection = document.getElementById("resultsSection");
  const resultContent = document.getElementById("resultContent");
  const extraBtns = document.getElementById("extraBtns");

  resultsSection.style.display = "block";
  extraBtns.style.display = actionType === "process" ? "flex" : "none";

  if (actionType === "process") {
    const highlightedOriginal = highlightWords(data.original, data.original_hard_words, "hard-word-highlight");
    const highlightedSimplified = highlightWords(data.simplified, data.survived_words, "hard-word-survived ");

    resultContent.innerHTML = `
    <div class="mb-4 p-3 rounded border" style="border-left: 5px solid var(--accent-color) !important;">
      <h6 class="text-uppercase mb-2" style="font-size:0.8rem;">Original · ${data.original_analysis.summary.reading_level}</h6>
      <p style="line-height:2;" id="original-text-block">${highlightedOriginal}</p>
    </div>
    <div class="p-3 rounded border" style="border-left: 5px solid var(--success-color) !important;">
      <h6 class="text-uppercase mb-2" style="font-size:0.8rem;">Simplified · ${data.simplified_analysis.summary.reading_level}</h6>
      <p style="line-height:2;">${highlightedSimplified}</p>
    </div>
  `;

    // Wire up tooltips
    const originalBlock = document.getElementById("original-text-block");
    initInteractiveWords(originalBlock);

  }

  if (actionType === "analyze") {
    const s = data.improvement_summary;
    resultContent.innerHTML = `
      <h5 class="border-bottom pb-2 mb-4">What Changed</h5>

      <p class="mb-4">${s.message}</p>

      <div class="row text-center mb-4">
        <div class="col">
          <div class="p-3 rounded border">
            <div class="h4 mb-0">${s.before_level}</div>
            <small>Before</small>
          </div>
        </div>
        <div class="col-auto d-flex align-items-center px-2">
          <i class="fa-solid fa-arrow-right"></i>
        </div>
        <div class="col">
          <div class="p-3 rounded border border-success">
            <div class="h4 mb-0 text-success">${s.after_level}</div>
            <small>After</small>
          </div>
        </div>
      </div>

      <div class="row text-center mb-4 g-3">
        <div class="col">
          <div class="p-3 rounded border">
            <div class="h4 text-success mb-0">${s.words_eliminated}</div>
            <small>Words Replaced</small>
          </div>
        </div>
        <div class="col">
          <div class="p-3 rounded border">
            <div class="h4 mb-0" style="color:var(--accent-color)">${s.words_survived}</div>
            <small>Still Hard</small>
          </div>
        </div>
        <div class="col">
          <div class="p-3 rounded border">
            <div class="h4 mb-0">${s.words_introduced}</div>
            <small>Introduced</small>
          </div>
        </div>
      </div>

      ${s.survived.length > 0 ? `
        <div class="mt-3">
          <h6>Words still to watch out for:</h6>
          ${s.survived.map(w => `
            <span class="badge bg-warning text-dark me-1 mb-1" title="${w.why}">
              ${w.word} · ${w.difficulty}
            </span>`).join('')}
        </div>` : ''}
        ${s.introduced.length > 0 ? `
        <div class="mt-3">
          ${s.introduced.map(w => `
            <span class="badge bg-warning text-dark me-1 mb-1" title="${w.why}">
              ${w.word} · ${w.difficulty}
            </span>`).join('')}
        </div>` : ''}
    `;
  }
}
// ── Contact Form ──
const contactCard = document.querySelector('.contact-card');
if (contactCard) observer.observe(contactCard);

document.getElementById('contactSubmit').addEventListener('click', () => {
  const name = document.getElementById('contactName').value.trim();
  const email = document.getElementById('contactEmail').value.trim();
  const subject = document.getElementById('contactSubject').value.trim();
  const message = document.getElementById('contactMessage').value.trim();

  if (!name || !email || !subject || !message) {
    alert('Please fill in all fields.');
    return;
  }

  document.getElementById('contactSuccess').style.display = 'block';

  // Clear form
  document.getElementById('contactName').value = '';
  document.getElementById('contactEmail').value = '';
  document.getElementById('contactSubject').value = '';
  document.getElementById('contactMessage').value = '';
});
//audio button
document.getElementById("volumeCheckbox").addEventListener("change", function () {
  const resultContent = document.getElementById("resultContent");
  if (this.checked) {
    const text = resultContent.innerText;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.85;
    speechSynthesis.speak(utterance);
  } else {
    speechSynthesis.cancel();
  }
});
//download button 
document.getElementById("downloadBtn").addEventListener("click", () => {
  const content = document.getElementById("resultContent").innerText;
  const blob = new Blob([content], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "dyslexia-result.txt";
  a.click();
});
//show of sparkle effect on the get started button
const btn = document.querySelector('.footer-btn');
let lastSparkle = 0;

btn.addEventListener('mousemove', (e) => {
  const now = Date.now();
  if (now - lastSparkle < 300) return;
  lastSparkle = now;

  const sparkle = document.createElement('span');
  sparkle.classList.add('sparkle');
  sparkle.textContent = ['✦', '★'][Math.floor(Math.random() * 2)];

  const rect = btn.getBoundingClientRect();
  sparkle.style.left = (e.clientX - rect.left + Math.random() * 20 - 10) + 'px';
  sparkle.style.top = (e.clientY - rect.top + Math.random() * 20 - 10) + 'px';

  btn.appendChild(sparkle);
  setTimeout(() => sparkle.remove(), 1000);

});
