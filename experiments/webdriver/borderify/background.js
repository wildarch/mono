browser.runtime.onMessage.addListener(async (msg) => {
    const res = await fetch("http://localhost:8000/experiments/webdriver/borderify/icons/border-48.png");
    console.log(res);
})
