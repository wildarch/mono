import browser from "webextension-polyfill";

function waitForSearchBox(limit: number) {
    const selector = 'div[aria-label="Search input textbox"]';
    const searchBox = document.querySelector(selector);
    // TODO: check if we have found it, otherwise wait a bit and retry.
}

addEventListener('mousemove', (event: MouseEvent) => {
    const selector = 'div[aria-label="Search input textbox"]';
    const target = document.querySelector(selector);
    if (target) {
        const targetRect = target.getBoundingClientRect();
        const mouseX = event.clientX;
        const mouseY = event.clientY;

        // Calculate the center of the targetRect
        const centerX = targetRect.left + targetRect.width / 2;
        const centerY = targetRect.top + targetRect.height / 2;

        let dx = 0;
        let dy = 0;
        const max_move = 10;

        // Calculate the distance from the mouse to the center of the target
        const diffX = mouseX - centerX;
        const diffY = mouseY - centerY;

        // Move towards the center if not already there, capping movement
        dx = -Math.sign(diffX) * Math.min(max_move, Math.abs(diffX));
        dy = -Math.sign(diffY) * Math.min(max_move, Math.abs(diffY));

        if (dx !== 0 || dy !== 0) {
            // Request mouse movement via background script
            browser.runtime.sendMessage({ x: Math.round(dx), y: Math.round(dy) });
        }
    }
});
