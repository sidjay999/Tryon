/**
 * Before/After comparison slider.
 * Uses clip-path on the "after" panel for smooth reveal.
 */

/**
 * @param {string} containerId - ID of the slider container
 * @param {string} handleId    - ID of the drag handle
 * @param {string} afterId     - ID of the "after" panel (clip-path applied here)
 */
export function initSlider(containerId, handleId, afterId) {
    const container = document.getElementById(containerId);
    const handle = document.getElementById(handleId);
    const afterPanel = document.getElementById(afterId);

    let dragging = false;
    let position = 50; // percent

    function setPosition(pct) {
        position = Math.max(2, Math.min(98, pct));
        afterPanel.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
        handle.style.left = `${position}%`;
    }

    function getPercent(clientX) {
        const rect = container.getBoundingClientRect();
        return ((clientX - rect.left) / rect.width) * 100;
    }

    // Mouse
    handle.addEventListener("mousedown", () => { dragging = true; });
    window.addEventListener("mousemove", (e) => { if (dragging) setPosition(getPercent(e.clientX)); });
    window.addEventListener("mouseup", () => { dragging = false; });

    // Touch
    handle.addEventListener("touchstart", (e) => { dragging = true; e.preventDefault(); }, { passive: false });
    window.addEventListener("touchmove", (e) => { if (dragging) setPosition(getPercent(e.touches[0].clientX)); }, { passive: true });
    window.addEventListener("touchend", () => { dragging = false; });

    // Also allow dragging anywhere on the container
    container.addEventListener("mousedown", (e) => { dragging = true; setPosition(getPercent(e.clientX)); });

    setPosition(50);
    return { setPosition };
}
