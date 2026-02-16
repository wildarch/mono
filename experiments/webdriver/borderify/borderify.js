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

        if (mouseX < targetRect.left) {
            console.log("need to move right");
        } else if (mouseX > targetRect.right) {
            console.log("need to move left");
        }

        if (mouseY < targetRect.top) {
            console.log("need to move down");
        } else if (mouseY > targetRect.bottom) {
            console.log("need to move up");
        }
    }
});
