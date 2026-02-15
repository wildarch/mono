document.body.style.border = "5px solid red";

browser.runtime.onMessage.addListener((msg) => {
    console.log(msg);
});

addEventListener('load', async () => {
    browser.runtime.sendMessage({});
});
