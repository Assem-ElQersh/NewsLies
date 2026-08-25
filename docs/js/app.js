import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let ortSession = null;
let tokenizer = null;
let TEMPERATURE = 1.0;

// The exact label mapping from the PyTorch model
const LABEL_CLASSES = ["credible", "not credible", "undecided"];

// Sample texts — real articles from the AFND dataset with high-confidence predictions
const SAMPLES = [
  // credible
  'تشاهدون في حلقة اليوم من "الوعد": البداية مع ريحان في الزنزانة تحكي للسجينات أيام البرد في قريتها كيف كانت، ومعلمتهم التي أخبرتهم أنهم يمكنهم مواجهة البرد بالعاطفة وكيف كان الأطفال يعانقون بعضهم عند الشعور بالبرد.',
  // not credible
  'رئيس الحكومة العراقية مصطفى الكاظمي اتخذ قراراً نهائياً بخصوص ضرائب رواتب الموظفين، وأصدر توجيهاً عاجلاً لوزارة المالية بإلغاء كتاب فرض الضرائب على الرواتب مؤكداً أن لا ضرائب ستُفرض على رواتب موظفي الدولة.',
  // undecided
  'وقّع رئيس الجمهورية، عبد المجيد تبون، مرسوما رئاسيا يتضمن تعديلا في الحكومة. وبمقتضى هذا المرسوم، عيّن رئيس الجمهورية، السيدات والسادة الآتية أسماؤهم: عبد العزيز جراد وزيرا أولا، صبري بوقدوم وزيرا للشؤون الخارجية.',
];

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
async function classify(text) {
  // Tokenize using Hugging Face WordPiece tokenizer (runs locally in browser)
  const tokenized = await tokenizer(text, {
    padding: 'max_length',
    truncation: true,
    max_length: 128,
  });

  // Extract int64 arrays
  const inputIdsData = tokenized.input_ids.data;
  const attentionMaskData = tokenized.attention_mask.data;

  // Build int64 tensors for ONNX [batch_size=1, max_seq_len=128]
  const inputTensor = new ort.Tensor("int64", inputIdsData, [1, 128]);
  const maskTensor = new ort.Tensor("int64", attentionMaskData, [1, 128]);

  // Run the AraBERT ONNX model
  const results = await ortSession.run({
    input_ids: inputTensor,
    attention_mask: maskTensor
  });

  const logits = results.logits.data; // Float32Array of length 3

  // Temperature-scaled softmax (temperature fitted on validation data only)
  const scaled = Array.from(logits, (x) => x / TEMPERATURE);
  const maxLogit = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExp);
}

function normalizedEntropy(probs) {
  const n = probs.length;
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return h / Math.log(n); // 0 = certain, 1 = maximally uncertain
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
  "credible":      { icon: "✅", label: "Credible",      cls: "credible"     },
  "not credible":  { icon: "❌", label: "Not Credible",  cls: "not-credible" },
  "undecided":     { icon: "⚠️", label: "Undecided",     cls: "undecided"    },
};

function showResult(probs) {
  const topIdx  = probs.indexOf(Math.max(...probs));
  const topClass = LABEL_CLASSES[topIdx];
  const m = META[topClass];

  document.getElementById("verdict-icon").textContent  = m.icon;
  const lbl = document.getElementById("verdict-label");
  lbl.textContent = m.label;
  lbl.className   = "verdict-label " + m.cls;
  document.getElementById("verdict-conf").textContent =
    `${(probs[topIdx] * 100).toFixed(1)}% calibrated confidence`;
  const unc = normalizedEntropy(probs);
  const uncLabel =
    unc < 0.25 ? "low uncertainty" :
    unc < 0.55 ? "moderate uncertainty" : "high uncertainty — treat with caution";
  document.getElementById("verdict-unc").textContent =
    `uncertainty ${unc.toFixed(2)} (${uncLabel})`;

  const barsEl = document.getElementById("bars");
  barsEl.innerHTML = "";
  LABEL_CLASSES.forEach((cls, i) => {
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
    LABEL_CLASSES.forEach((_, i) => {
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
  setStatus("Downloading model…", "loading");
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

    // 0. Optional calibration temperature (fitted on validation data only)
    try {
      const calib = await fetch("calibration.json");
      if (calib.ok) {
        const j = await calib.json();
        if (typeof j.temperature === "number") TEMPERATURE = j.temperature;
      }
    } catch (_) { /* default T=1.0 */ }

    // 1. Load Hugging Face Tokenizer (AraBERT WordPiece)
    tokenizer = await AutoTokenizer.from_pretrained('AZZOMA/newslies');

    // 2. Load ONNX Model from Hugging Face Hub (Fetches .onnx and .onnx.data)
    ortSession = await ort.InferenceSession.create(
      "https://huggingface.co/AZZOMA/newslies/resolve/main/model.onnx", 
      { executionProviders: ["wasm"] }
    );

    setStatus(TEMPERATURE !== 1.0 ? "Ready · calibrated" : "Ready", "ready");
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
  setStatus("Classifying...", "loading");

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

document.getElementById("input-text").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    document.getElementById("classify-btn").click();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadResources();
