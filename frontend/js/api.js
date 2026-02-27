/**
 * API client – Phase 2 (synchronous response).
 * The API now returns the result directly in the POST response.
 * No polling needed.
 */

const API_BASE = "";  // empty = same origin

/**
 * Submit a try-on job and wait for the result directly.
 * @param {File} personFile
 * @param {File} clothingFile
 * @param {string} garmentCategory - "upper" | "full" | "lower"
 * @param {(label: string) => void} onStep - called with step labels during wait
 * @returns {Promise<{result_url?: string, result_b64?: string, job_id: string}>}
 */
export async function submitTryOn(personFile, clothingFile, garmentCategory = "upper", onStep) {
    const form = new FormData();
    form.append("person_image", personFile);
    form.append("clothing_image", clothingFile);
    form.append("garment_category", garmentCategory);

    // Simulate step labels while the server is processing
    const steps = [
        "Segmenting body…",
        "Extracting pose…",
        "Warping clothing…",
        "Generating with SDXL…",
        "Blending & finishing…",
    ];
    let stepIdx = 0;
    const stepTimer = setInterval(() => {
        if (stepIdx < steps.length) {
            onStep?.(steps[stepIdx++]);
        }
    }, 3000);

    try {
        const res = await fetch(`${API_BASE}/api/tryon`, {
            method: "POST",
            body: form,
        });

        clearInterval(stepTimer);

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const detail = err.detail;
            if (typeof detail === "object") throw new Error(detail.error || "Server error");
            throw new Error(detail || `Server error ${res.status}`);
        }

        const data = await res.json();
        return data;
    } catch (err) {
        clearInterval(stepTimer);
        throw err;
    }
}
