/**
 * API client – communicates with the FastAPI backend.
 */

const API_BASE = "";   // empty = same origin; set to "http://localhost:8000" for local dev

/**
 * Submit a try-on job.
 * @param {File} personFile
 * @param {File} clothingFile
 * @returns {Promise<string>} job_id
 */
export async function submitTryOn(personFile, clothingFile) {
    const form = new FormData();
    form.append("person_image", personFile);
    form.append("clothing_image", clothingFile);

    const res = await fetch(`${API_BASE}/api/tryon`, {
        method: "POST",
        body: form,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    return data.job_id;
}

/**
 * Poll job status until complete or failed.
 * @param {string} jobId
 * @param {(meta: {step:string, progress:number}) => void} onProgress
 * @returns {Promise<{result_url?:string, result_b64?:string}>}
 */
export async function pollJobStatus(jobId, onProgress) {
    const INTERVAL = 1500;
    const TIMEOUT = 120_000;
    const started = Date.now();

    return new Promise((resolve, reject) => {
        const tick = async () => {
            if (Date.now() - started > TIMEOUT) {
                reject(new Error("Job timed out after 2 minutes"));
                return;
            }

            try {
                const res = await fetch(`${API_BASE}/api/tryon/${jobId}`);
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    reject(new Error(err.detail?.error || `Server error ${res.status}`));
                    return;
                }

                const data = await res.json();

                if (data.status === "completed") {
                    resolve(data);
                    return;
                }

                if (data.status === "failed" || data.status === "failure") {
                    reject(new Error(data.error || "Pipeline failed"));
                    return;
                }

                // Progress update
                onProgress?.({
                    step: data.step || data.status,
                    progress: data.progress || 0,
                });

                setTimeout(tick, INTERVAL);
            } catch (err) {
                reject(err);
            }
        };

        setTimeout(tick, INTERVAL);
    });
}
