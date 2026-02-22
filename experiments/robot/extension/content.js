document.body.style.border = "5px solid red";

browser.runtime.onMessage.addListener((msg) => {
    console.log(msg);
});

addEventListener('load', async () => {
    browser.runtime.sendMessage({});
});

addEventListener('mousemove', (event) => {
    const target = document.querySelector('#trying_it_out');
    if (target) {
        const targetRect = target.firstElementChild.getBoundingClientRect();
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
            browser.runtime.sendMessage({ x: Math.round(dx), y: Math.round(dy) });
        }
    }
});
