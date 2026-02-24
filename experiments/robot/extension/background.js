browser.runtime.onMessage.addListener(async (msg) => {
    try {
        await fetch("http://localhost:8000/", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(msg),
        });
    } catch (error) {
        console.error("Error sending input request to input server:", error);
    }
});
