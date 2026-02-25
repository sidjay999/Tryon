/**
 * Drag-and-drop upload zones.
 */

/**
 * @typedef {Object} UploaderOptions
 * @property {string} zoneId   - ID of the drop zone element
 * @property {string} inputId  - ID of the <input type="file"> element
 * @property {string} previewWrapperId - ID of the preview wrapper
 * @property {string} previewImgId    - ID of the <img> inside the preview wrapper
 * @property {string} removeId        - ID of the remove button
 * @property {(file: File|null) => void} onChange - called whenever file changes
 */

/**
 * @param {UploaderOptions} opts
 */
export function initUploader(opts) {
    const zone = document.getElementById(opts.zoneId);
    const input = document.getElementById(opts.inputId);
    const previewWrap = document.getElementById(opts.previewWrapperId);
    const previewImg = document.getElementById(opts.previewImgId);
    const removeBtn = document.getElementById(opts.removeId);

    let currentFile = null;

    function setFile(file) {
        if (!file || !file.type.startsWith("image/")) return;
        if (file.size > 20 * 1024 * 1024) {
            alert("Image must be under 20MB");
            return;
        }
        currentFile = file;
        const url = URL.createObjectURL(file);
        previewImg.src = url;
        previewWrap.classList.remove("hidden");
        zone.querySelector(".upload-zone-inner").classList.add("hidden");
        opts.onChange(currentFile);
    }

    function clearFile() {
        currentFile = null;
        previewImg.src = "";
        previewWrap.classList.add("hidden");
        zone.querySelector(".upload-zone-inner").classList.remove("hidden");
        input.value = "";
        opts.onChange(null);
    }

    // Click to pick
    zone.addEventListener("click", (e) => {
        if (e.target === removeBtn || removeBtn.contains(e.target)) return;
        if (currentFile) return;
        input.click();
    });

    input.addEventListener("change", () => {
        if (input.files[0]) setFile(input.files[0]);
    });

    // Drag and drop
    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("drag-over");
        const file = e.dataTransfer?.files?.[0];
        if (file) setFile(file);
    });

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFile();
    });

    return {
        getFile: () => currentFile,
        clear: clearFile,
    };
}
