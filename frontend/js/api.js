/**
 * API client – CatVTON Pipeline (synchronous response).
 * The API returns the result directly in the POST response.
 */

const API_BASE = "";  // empty = same origin

/**
 * Submit a try-on job and wait for the result directly.
 * @param {File} personFile
 * @param {File} clothingFile
 * @param {string} garmentCategory - "upper" | "lower" | "overall"
 * @param {(label: string) => void} onStep - called with step labels during wait
 * @returns {Promise<{result_image_base64?: string, job_id: string}>}
 */
export async function submitTryOn(personFile, clothingFile, garmentCategory = "upper", onStep) {
    const form = new FormData();
    form.append("person_image", personFile);
    form.append("clothing_image", clothingFile);
    form.append("garment_category", garmentCategory);

    // Simulate step labels while the server is processing
    const steps = [
        "Parsing body with DensePose + SCHP…",
        "Running CatVTON diffusion…",
        "Finishing up…",
    ];
    let stepIdx = 0;
    const stepTimer = setInterval(() => {
        if (stepIdx < steps.length) {
            onStep?.(steps[stepIdx++]);
        }
    }, 8000);  // ~8s per step for 50-step diffusion

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
