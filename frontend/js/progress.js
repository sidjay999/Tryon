/**
 * Progress indicator – animates pipeline step cards and the progress bar.
 */

const STEP_ORDER = ["segmentation", "pose", "warp", "inpainting", "blend"];

const STEP_LABELS = {
    segmentation: "Running human segmentation…",
    pose: "Extracting pose keypoints…",
    warp: "Warping clothing to body…",
    inpainting: "Generating with SDXL…",
    blend: "Blending & finishing…",
    done: "Done!",
};

export function showProgress() {
    document.getElementById("progress-section").classList.remove("hidden");
    document.getElementById("result-section").classList.add("hidden");
    document.getElementById("error-section").classList.add("hidden");
    setProgress({ step: "segmentation", progress: 5 });
}

export function hideProgress() {
    document.getElementById("progress-section").classList.add("hidden");
}

/**
 * @param {{ step: string, progress: number }} meta
 */
export function setProgress({ step, progress }) {
    // Progress bar
    const bar = document.getElementById("progress-bar");
    bar.style.width = `${Math.min(progress, 100)}%`;

    // Label
    const label = document.getElementById("progress-label");
    label.textContent = STEP_LABELS[step] || `${step}…`;

    // Step cards
    const stepIdx = STEP_ORDER.indexOf(step);
    document.querySelectorAll(".progress-step").forEach((el, i) => {
        el.classList.remove("active", "done");
        if (i < stepIdx) el.classList.add("done");
        else if (i === stepIdx) el.classList.add("active");
    });
}
