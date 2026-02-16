browser.runtime.onMessage.addListener(async (msg) => {
    if (msg.x !== undefined && msg.y !== undefined) {
        try {
            await fetch("http://localhost:8000/", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(msg),
            });
        } catch (error) {
            console.error("Error sending mouse movement to input server:", error);
        }
    }
});
