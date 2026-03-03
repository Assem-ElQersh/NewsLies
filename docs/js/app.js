"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let ortSession  = null;
let vocab       = null;   // { word2idx, label_classes, max_seq_len }
let stopWordSet = null;   // Set<string>
const stemmer   = new ISRIStemmer();

// Sample texts — real articles from the AFND dataset with high-confidence predictions
const SAMPLES = [
  // credible (~76% confidence)
  'تشاهدون في حلقة اليوم من "الوعد": البداية مع ريحان في الزنزانة تحكي للسجينات أيام البرد في قريتها كيف كانت، ومعلمتهم التي أخبرتهم أنهم يمكنهم مواجهة البرد بالعاطفة وكيف كان الأطفال يعانقون بعضهم عند الشعور بالبرد.',
  // not credible (~90% confidence)
  'وقّع رئيس الجمهورية، عبد المجيد تبون، مرسوما رئاسيا يتضمن تعديلا في الحكومة. وبمقتضى هذا المرسوم، عيّن رئيس الجمهورية، السيدات والسادة الآتية أسماؤهم: عبد العزيز جراد وزيرا أولا، صبري بوقدوم وزيرا للشؤون الخارجية.',
  // undecided (~97% confidence)
  'رئيس الحكومة العراقية مصطفى الكاظمي اتخذ قراراً نهائياً بخصوص ضرائب رواتب الموظفين، وأصدر توجيهاً عاجلاً لوزارة المالية بإلغاء كتاب فرض الضرائب على الرواتب مؤكداً أن لا ضرائب ستُفرض على رواتب موظفي الدولة.',
];

// ---------------------------------------------------------------------------
// ISRIStemmer — complete JavaScript port of NLTK's nltk.stem.isri.ISRIStemmer
// ---------------------------------------------------------------------------
function ISRIStemmer() {
  // Length-3 prefixes
  const p3 = ["\u0643\u0627\u0644", "\u0628\u0627\u0644", "\u0648\u0644\u0644", "\u0648\u0627\u0644"]; // كال،بال،ولل،وال
  // Length-2 prefixes
  const p2 = ["\u0627\u0644", "\u0644\u0644"]; // ال،لل
  // Length-1 prefixes
  const p1 = ["\u0644", "\u0628", "\u0641", "\u0633", "\u0648", "\u064a", "\u062a", "\u0646", "\u0627"]; // ل،ب،ف،س،و،ي،ت،ن،ا

  // Length-3 suffixes
  const s3 = ["\u062a\u0645\u0644", "\u0647\u0645\u0644", "\u062a\u0627\u0646", "\u062a\u064a\u0646", "\u0643\u0645\u0644"]; // تمل،همل،تان،تين،كمل
  // Length-2 suffixes
  const s2 = ["\u0648\u0646","\u0627\u062a","\u0627\u0646","\u064a\u0646","\u062a\u0646","\u0643\u0645","\u0647\u0646","\u0646\u0627","\u064a\u0627","\u0647\u0627","\u062a\u0645","\u0643\u0646","\u0646\u064a","\u0648\u0627","\u0645\u0627","\u0647\u0645"]; // ون،ات،ان،ين،تن،كم،هن،نا،يا،ها،تم،كن،ني،وا،ما،هم
  // Length-1 suffixes
  const s1 = ["\u0629", "\u0647", "\u064a", "\u0643", "\u062a", "\u0627", "\u0646"]; // ة،ه،ي،ك،ت،ا،ن

  // pr4: patterns for length-4 word → length-3 root
  const pr4 = {
    0: ["\u0645"],                              // م
    1: ["\u0627"],                              // ا
    2: ["\u0627", "\u0648", "\u064a"],          // ا،و،ي
    3: ["\u0629"],                              // ة
  };

  // pr53: patterns for length-5 word → length-3 root
  const pr53 = {
    0: ["\u0627", "\u062a"],                    // ا،ت
    1: ["\u0627", "\u064a", "\u0648"],          // ا،ي،و
    2: ["\u0627", "\u062a", "\u0645"],          // ا،ت،م
    3: ["\u0645", "\u064a", "\u062a"],          // م،ي،ت
    4: ["\u0645", "\u062a"],                    // م،ت
    5: ["\u0627", "\u0648"],                    // ا،و
    6: ["\u0627", "\u0645"],                    // ا،م
  };

  // Stemmer's own stop words (returned unchanged)
  const stopWords = new Set(["\u064a\u0643\u0648\u0646","\u0648\u0644\u064a\u0633","\u0648\u0643\u0627\u0646","\u0643\u0630\u0644\u0643","\u0627\u0644\u062a\u064a","\u0648\u0628\u064a\u0646","\u0639\u0644\u064a\u0647\u0627","\u0645\u0633\u0627\u0621","\u0627\u0644\u0630\u064a","\u0648\u0643\u0627\u0646\u062a","\u0648\u0644\u0643\u0646","\u0648\u0627\u0644\u062a\u064a","\u062a\u0643\u0648\u0646","\u0627\u0644\u064a\u0648\u0645","\u0627\u0644\u0644\u0630\u064a\u0646","\u0639\u0644\u064a\u0647","\u0643\u0627\u0646\u062a","\u0644\u0630\u0644\u0643","\u0623\u0645\u0627\u0645","\u0647\u0646\u0627\u0643","\u0645\u0646\u0647\u0627","\u0645\u0627\u0632\u0627\u0644","\u0644\u0627\u0632\u0627\u0644","\u0644\u0627\u064a\u0632\u0627\u0644","\u0645\u0627\u064a\u0632\u0627\u0644","\u0627\u0635\u0628\u062d","\u0623\u0635\u0628\u062d","\u0623\u0645\u0633\u0649","\u0627\u0645\u0633\u0649","\u0623\u0636\u062d\u0649","\u0627\u0636\u062d\u0649","\u0645\u0627\u0628\u0631\u062d","\u0645\u0627\u0641\u062a\u0626","\u0645\u0627\u0627\u0646\u0641\u0643","\u0644\u0627\u0633\u064a\u0645\u0627","\u0648\u0644\u0627\u064a\u0632\u0627\u0644","\u0627\u0644\u062d\u0627\u0644\u064a","\u0627\u0644\u064a\u0647\u0627","\u0627\u0644\u0630\u064a\u0646","\u0641\u0627\u0646\u0647","\u0648\u0627\u0644\u0630\u064a","\u0648\u0647\u0630\u0627","\u0644\u0647\u0630\u0627","\u0641\u0643\u0627\u0646","\u0633\u062a\u0643\u0648\u0646","\u0627\u0644\u064a\u0647","\u064a\u0645\u0643\u0646","\u0628\u0647\u0630\u0627","\u0627\u0644\u0630\u0649"]);

  // Regex for Arabic short vowels (diacritics) U+064B–U+0652
  const reShortVowels    = /[\u064B-\u0652]/g;
  // Regex for initial hamza forms (أ، إ، آ) — normalize to bare alef ا
  const reInitialHamza   = /^[\u0622\u0623\u0625]/;

  function norm1(w) { return w.replace(reShortVowels, ""); }
  function norm2(w) { return w.replace(reInitialHamza, "\u0627"); }

  function pre32(w) {
    if (w.length >= 6) { for (const p of p3) if (w.startsWith(p)) return w.slice(3); }
    if (w.length >= 5) { for (const p of p2) if (w.startsWith(p)) return w.slice(2); }
    return w;
  }

  function suf32(w) {
    if (w.length >= 6) { for (const s of s3) if (w.endsWith(s)) return w.slice(0, -3); }
    if (w.length >= 5) { for (const s of s2) if (w.endsWith(s)) return w.slice(0, -2); }
    return w;
  }

  function waw(w) {
    if (w.length >= 4 && w[0] === "\u0648" && w[1] === "\u0648") return w.slice(1);
    return w;
  }

  function suf1(w) {
    for (const s of s1) if (w.endsWith(s)) return w.slice(0, -1);
    return w;
  }

  function pre1(w) {
    for (const p of p1) if (w.startsWith(p)) return w.slice(1);
    return w;
  }

  function proW4(w) {
    if (pr4[0].includes(w[0])) return w.slice(1);
    if (pr4[1].includes(w[1])) return w[0] + w.slice(2);
    if (pr4[2].includes(w[2])) return w.slice(0, 2) + w[3];
    if (pr4[3].includes(w[3])) return w.slice(0, -1);
    w = suf1(w);
    if (w.length === 4) w = pre1(w);
    return w;
  }

  function proW53(w) {
    if (pr53[0].includes(w[2]) && w[0] === "\u0627") return w[1] + w.slice(3);
    if (pr53[1].includes(w[3]) && w[0] === "\u0645") return w.slice(1, 3) + w[4];
    if (pr53[2].includes(w[0]) && w[4] === "\u0629") return w.slice(1, 4);
    if (pr53[3].includes(w[0]) && w[2] === "\u062a") return w[1] + w.slice(3);
    if (pr53[4].includes(w[0]) && w[2] === "\u0627") return w[1] + w.slice(3);
    if (pr53[5].includes(w[2]) && w[4] === "\u0629") return w.slice(0, 2) + w[3];
    if (pr53[6].includes(w[0]) && w[1] === "\u0646") return w.slice(2);
    if (w[3] === "\u0627" && w[0] === "\u0627") return w.slice(1, 3) + w[4];
    if (w[4] === "\u0646" && w[3] === "\u0627") return w.slice(0, 3);
    if (w[3] === "\u064a" && w[0] === "\u062a") return w.slice(1, 3) + w[4];
    if (w[3] === "\u0648" && w[1] === "\u0627") return w[0] + w[2] + w[4];
    if (w[2] === "\u0627" && w[1] === "\u0648") return w[0] + w.slice(3);
    if (w[3] === "\u0626" && w[2] === "\u0627") return w.slice(0, 2) + w[4];
    if (w[4] === "\u0629" && w[1] === "\u0627") return w[0] + w.slice(2, 4);
    if (w[4] === "\u064a" && w[2] === "\u0627") return w.slice(0, 2) + w[3];
    w = suf1(w);
    if (w.length === 5) w = pre1(w);
    return w;
  }

  function proW54(w) {
    if (pr53[2].includes(w[0])) return w.slice(1);
    if (w[4] === "\u0629") return w.slice(0, 4);
    if (w[2] === "\u0627") return w.slice(0, 2) + w.slice(3);
    return w;
  }

  function endW5(w) {
    if (w.length === 4) return proW4(w);
    if (w.length === 5) return proW54(w);
    return w;
  }

  function proW6(w) {
    if (w.startsWith("\u0627\u0633\u062a") || w.startsWith("\u0645\u0633\u062a")) return w.slice(3);
    if (w[0] === "\u0645" && w[3] === "\u0627" && w[5] === "\u0629") return w.slice(1, 3) + w[4];
    if (w[0] === "\u0627" && w[2] === "\u062a" && w[4] === "\u0627") return w[1] + w[3] + w[5];
    if (w[0] === "\u0627" && w[3] === "\u0648" && w[2] === w[4])     return w[1] + w.slice(4);
    if (w[0] === "\u062a" && w[2] === "\u0627" && w[4] === "\u064a") return w[1] + w[3] + w[5];
    w = suf1(w);
    if (w.length === 6) w = pre1(w);
    return w;
  }

  function proW64(w) {
    if (w[0] === "\u0627" && w[4] === "\u0627") return w.slice(1, 4) + w[5];
    if (w.startsWith("\u0645\u062a"))            return w.slice(2);
    return w;
  }

  function endW6(w) {
    if (w.length === 5) { w = proW53(w); w = endW5(w); }
    else if (w.length === 6) { w = proW64(w); }
    return w;
  }

  this.stem = function (token) {
    token = norm1(token);
    if (stopWords.has(token)) return token;
    token = pre32(token);
    token = suf32(token);
    token = waw(token);
    token = norm2(token);
    const len = token.length;
    if      (len === 4) token = proW4(token);
    else if (len === 5) { token = proW53(token); token = endW5(token); }
    else if (len === 6) { token = proW6(token);  token = endW6(token); }
    else if (len === 7) {
      token = suf1(token);
      if (token.length === 7) token = pre1(token);
      if (token.length === 6) { token = proW6(token); token = endW6(token); }
    }
    return token;
  };
}

// ---------------------------------------------------------------------------
// Text preprocessing  (mirrors train_pytorch.py pipeline)
// ---------------------------------------------------------------------------
function preprocess(text) {
  const tokens = text
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0 && !stopWordSet.has(w))
    .map((w) => stemmer.stem(w));
  return tokens;
}

function tokenizeAndPad(tokens) {
  const { word2idx, max_seq_len } = vocab;
  const ids = new Array(max_seq_len).fill(0); // 0 = padding
  const len = Math.min(tokens.length, max_seq_len);
  for (let i = 0; i < len; i++) {
    ids[i] = word2idx[tokens[i]] ?? 0;
  }
  return ids;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
async function classify(text) {
  const tokens = preprocess(text);
  const ids    = tokenizeAndPad(tokens);

  // Build int64 tensor  [1, max_seq_len]
  const inputTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(ids.map(BigInt)),
    [1, vocab.max_seq_len]
  );

  const results = await ortSession.run({ input_ids: inputTensor });
  const logits  = results.logits.data; // Float32Array of length 3

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = Array.from(logits).map((x) => Math.exp(x - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExp);
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function setStatus(text, state) {
  document.getElementById("status-text").textContent = text;
  const dot = document.getElementById("status-dot");
  dot.className = "dot " + state; // loading | ready | error
}

const META = {
  credible:      { icon: "✅", label: "Credible",      cls: "credible"     },
  "not credible":{ icon: "❌", label: "Not Credible",  cls: "not-credible" },
  undecided:     { icon: "⚠️", label: "Undecided",     cls: "undecided"    },
};

function showResult(probs) {
  const classes = vocab.label_classes;
  const topIdx  = probs.indexOf(Math.max(...probs));
  const topClass = classes[topIdx];
  const m = META[topClass];

  document.getElementById("verdict-icon").textContent  = m.icon;
  const lbl = document.getElementById("verdict-label");
  lbl.textContent = m.label;
  lbl.className   = "verdict-label " + m.cls;
  document.getElementById("verdict-conf").textContent =
    `${(probs[topIdx] * 100).toFixed(1)}% confidence`;

  const barsEl = document.getElementById("bars");
  barsEl.innerHTML = "";
  classes.forEach((cls, i) => {
    const pct  = (probs[i] * 100).toFixed(1);
    const meta = META[cls];
    const row  = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-header">
        <span class="bar-name">${meta.icon} ${meta.label}</span>
        <span class="bar-pct">${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${meta.cls}" id="bar-${i}"></div>
      </div>`;
    barsEl.appendChild(row);
  });

  // Animate bars after a tick so CSS transition fires
  requestAnimationFrame(() => {
    classes.forEach((_, i) => {
      const pct = (probs[i] * 100).toFixed(1);
      document.getElementById(`bar-${i}`).style.width = pct + "%";
    });
  });

  const card = document.getElementById("results-card");
  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Resource loading
// ---------------------------------------------------------------------------
async function loadResources() {
  setStatus("Loading model…", "loading");
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

    const [session, vocabResp, swResp] = await Promise.all([
      ort.InferenceSession.create("model.onnx", {
        executionProviders: ["wasm"],
      }),
      fetch("vocab.json").then((r) => r.json()),
      fetch("stopwords.json").then((r) => r.json()),
    ]);

    ortSession  = session;
    vocab       = vocabResp;
    stopWordSet = new Set(swResp);

    setStatus("Ready", "ready");
    document.getElementById("classify-btn").disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("Failed to load model", "error");
  }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
document.getElementById("classify-btn").addEventListener("click", async () => {
  const text = document.getElementById("input-text").value.trim();
  if (!text) return;

  const btn = document.getElementById("classify-btn");
  btn.disabled = true;
  setStatus("Classifying…", "loading");

  try {
    const probs = await classify(text);
    showResult(probs);
    setStatus("Ready", "ready");
  } catch (err) {
    console.error(err);
    setStatus("Error during inference", "error");
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("clear-btn").addEventListener("click", () => {
  document.getElementById("input-text").value = "";
  document.getElementById("results-card").style.display = "none";
});

document.querySelectorAll(".sample-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("input-text").value = SAMPLES[+btn.dataset.idx];
    document.getElementById("results-card").style.display = "none";
  });
});

// Also allow Ctrl+Enter to classify
document.getElementById("input-text").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    document.getElementById("classify-btn").click();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadResources();
