/**
 * Main app – Phase 2 (synchronous API).
 * Result comes back in a single POST response — no polling.
 */
import { submitTryOn } from "./api.js";
import { initUploader } from "./uploader.js";
import { showProgress, hideProgress, setProgress } from "./progress.js";
import { initSlider } from "./slider.js";

// ── Elements ──────────────────────────────────────────────────
const btnGenerate = document.getElementById("btn-generate");
const btnDownload = document.getElementById("btn-download");
const btnReset = document.getElementById("btn-reset");
const btnRetry = document.getElementById("btn-retry");
const submitNote = document.getElementById("submit-note");
const resultSection = document.getElementById("result-section");
const errorSection = document.getElementById("error-section");
const errorMsg = document.getElementById("error-msg");
const resultBefore = document.getElementById("result-before");
const resultAfter = document.getElementById("result-after");
const sliderContainer = document.getElementById("slider-container");

// ── State ─────────────────────────────────────────────────────
let personFile = null;
let clothingFile = null;
let resultDataUrl = null;
let sliderInstance = null;
let personPreviewUrl = null;
let stepCounter = 0;

// ── Uploaders ─────────────────────────────────────────────────
const personUploader = initUploader({
    zoneId: "zone-person", inputId: "input-person",
    previewWrapperId: "preview-person", previewImgId: "preview-person-img",
    removeId: "remove-person",
    onChange(file) { personFile = file; personPreviewUrl = file ? URL.createObjectURL(file) : null; updateBtn(); },
});

const clothingUploader = initUploader({
    zoneId: "zone-clothing", inputId: "input-clothing",
    previewWrapperId: "preview-clothing", previewImgId: "preview-clothing-img",
    removeId: "remove-clothing",
    onChange(file) { clothingFile = file; updateBtn(); },
});

function updateBtn() {
    const ready = personFile && clothingFile;
    btnGenerate.disabled = !ready;
    submitNote.textContent = ready ? "Both images loaded — ready to generate!" : "Upload both images to generate";
}

// ── Generate ──────────────────────────────────────────────────
async function generate() {
    setUIState("loading");
    showProgress();
    stepCounter = 0;

    // Map step labels from api.js to progress values
    const STEP_PROGRESS = {
        "Segmenting body…": { step: "segmentation", progress: 15 },
        "Extracting pose…": { step: "pose", progress: 30 },
        "Warping clothing…": { step: "warp", progress: 48 },
        "Generating with SDXL…": { step: "inpainting", progress: 65 },
        "Blending & finishing…": { step: "blend", progress: 88 },
    };

    try {
        const result = await submitTryOn(
            personFile,
            clothingFile,
            "upper",                    // garment_category — expandable later
            (label) => {
                const p = STEP_PROGRESS[label];
                if (p) setProgress(p);
            },
        );

        if (result.result_url) resultDataUrl = result.result_url;
        else if (result.result_b64) resultDataUrl = `data:image/png;base64,${result.result_b64}`;
        else throw new Error("No result image returned");

        showResult(resultDataUrl);
    } catch (err) {
        showError(err.message || "Pipeline failed. Please try again.");
    }
}

// ── Show Result ───────────────────────────────────────────────
function showResult(afterUrl) {
    hideProgress();
    resultBefore.src = personPreviewUrl;
    resultAfter.src = afterUrl;
    resultBefore.onload = () => {
        sliderContainer.style.aspectRatio = `${resultBefore.naturalWidth}/${resultBefore.naturalHeight}`;
    };
    resultSection.classList.remove("hidden");
    errorSection.classList.add("hidden");
    if (!sliderInstance) sliderInstance = initSlider("slider-container", "slider-handle", "slider-after");
    else sliderInstance.setPosition(50);
    setUIState("done");
}

function showError(msg) {
    hideProgress();
    errorMsg.textContent = msg;
    errorSection.classList.remove("hidden");
    resultSection.classList.add("hidden");
    setUIState("idle");
}

function setUIState(state) {
    const btnText = btnGenerate.querySelector(".btn-text");
    const spinner = btnGenerate.querySelector(".btn-spinner");
    if (state === "loading") {
        btnGenerate.disabled = true;
        btnText.classList.add("hidden");
        spinner.classList.remove("hidden");
    } else {
        btnText.classList.remove("hidden");
        spinner.classList.add("hidden");
        if (state === "idle") btnGenerate.disabled = !(personFile && clothingFile);
    }
}

// ── Download ──────────────────────────────────────────────────
btnDownload.addEventListener("click", async () => {
    if (!resultDataUrl) return;
    if (resultDataUrl.startsWith("data:")) {
        const a = document.createElement("a"); a.href = resultDataUrl; a.download = "tryon-result.png"; a.click();
    } else {
        const blob = await fetch(resultDataUrl).then(r => r.blob());
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a"); a.href = url; a.download = "tryon-result.png"; a.click();
        URL.revokeObjectURL(url);
    }
});

// ── Reset ─────────────────────────────────────────────────────
function reset() {
    personUploader.clear(); clothingUploader.clear();
    personFile = null; clothingFile = null; resultDataUrl = null; personPreviewUrl = null;
    hideProgress();
    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    setUIState("idle"); updateBtn();
}
btnReset.addEventListener("click", reset);
btnRetry.addEventListener("click", () => { errorSection.classList.add("hidden"); generate(); });
btnGenerate.addEventListener("click", generate);

// ── Background particles ──────────────────────────────────────
(function spawnParticles() {
    const container = document.getElementById("bg-particles");
    if (!container) return;
    for (let i = 0; i < 18; i++) {
        const p = document.createElement("div"); p.className = "particle";
        const size = Math.random() * 4 + 2;
        p.style.cssText = `width:${size}px;height:${size}px;left:${Math.random() * 100}%;animation-duration:${8 + Math.random() * 14}s;animation-delay:${Math.random() * -20}s;opacity:${0.2 + Math.random() * 0.5};`;
        container.appendChild(p);
    }
})();
