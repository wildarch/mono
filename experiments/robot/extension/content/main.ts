import browser from "webextension-polyfill";

export interface InputRequest {
    mouseMoveRel?: {
        x: number,
        y: number,
    }
    mouseClickLeft?: boolean,
    mouseClickRight?: boolean,
}

function sendInputRequest(req: InputRequest) {
    browser.runtime.sendMessage(req);
}

async function delay(millis: number) {
    return new Promise(resolve => setTimeout(resolve, millis));
}

async function waitForSearchBox() {
    const selector = 'div[aria-label="Search input textbox"]';
    let searchBox = document.querySelector(selector);
    // TODO: Don't wait forever
    while (!searchBox) {
        await delay(100);
        searchBox = document.querySelector(selector);
    }

    return searchBox;
}

async function moveTo(target: Element) {
    mouseTarget = target;
    // Move a little bit to trigger mousemove loop.
    sendInputRequest({
        mouseMoveRel: {
            x: 10,
            y: 10
        }
    });
    return new Promise<void>(resolve => {
        target.addEventListener(
            'mouseenter',
            () => {
                mouseTarget = null;
                resolve();
            },
            { once: true });
    });
}

let mouseTarget: Element | null = null;
addEventListener('mousemove', (event: MouseEvent) => {
    if (mouseTarget) {
        const targetRect = mouseTarget.getBoundingClientRect();
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
            sendInputRequest({
                mouseMoveRel: {
                    x: Math.round(dx),
                    y: Math.round(dy),
                }
            })
        }
    }
});

async function main() {
    console.log("Looking for search box");
    const searchBox = await waitForSearchBox();
    console.log("Found: ", searchBox);
    await moveTo(searchBox);
    console.log("At search box!: ");
    await delay(100);
    sendInputRequest({ mouseClickLeft: true });
    await delay(100);
    sendInputRequest({ mouseClickLeft: true });
    await delay(100);
    sendInputRequest({ mouseClickLeft: true });
    await delay(100);
    sendInputRequest({ mouseClickLeft: true });
    await delay(100);
    sendInputRequest({ mouseClickLeft: true });
}

main();
