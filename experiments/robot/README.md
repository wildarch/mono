# APIs op top of WebDriver
Many services that I use every day do not have a public API, forcing me to use a web interface to interact with them.
However, because Firefox and Chrome can be controlled programmatically using the WebDriver protocol, I still think it is possible to build the APIs that I want, using the browser as an intermediary.

As an initial project, I want to build an API around WhatsApp Web.

## Avoiding Bot Detection
It seems that some websites try to [detect usage of the WebDriver protocol](https://stackoverflow.com/questions/73662698/check-if-selenium-webdriver-is-being-detected) to protect against bots scraping their content.
If I am not careful, using WebDriver on such websites may get my account banned.
Testing tools:
- https://deviceandbrowserinfo.com/are_you_a_bot
- https://www.browserscan.net/bot-detection

It appears that  is a pretty safe option to use.

## Attaching to Existing Window
I start Firefox from the command line as `firefox --marionette`.
This lets me use Firefox as normal, but I can also attach a WebDriver script to it.
I immediately see issues with this approach:
- Firefox displays a robot icon in the address bar, and an annoying striping pattern.
- It trivially fails bot detection methods.

This is clearly not a viable approach.
There are some selenium forks such as [SeleniumBase](https://github.com/seleniumbase/SeleniumBase) that claim to bypass all this, but they don't support attaching to an existing browser session as far as I can see.

## Using an Extension
Firefox natively supports extensions.
It should be pretty easy to develop an extension that can inspect the pages I want to monitor, without being detected by the usual bot detection tools.
Extensions can also provide JavaScript to be loaded into the page, so I can programmatically inspect the DOM quite easily.
I followed [the tutorial by Mozilla](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Your_first_WebExtension), and was able to inject custom javascript into the WhatsApp Web page.

The major downside is that extensions cannot simulate input events like WebDriver can, and we need this to drive more advanced APIs.

## Programming Input Events
It seems to most popular library for this is [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/).
Unfortunately it does not work with wayland, so I cannot use it.

I had some success with `python3-uinput`.
After installing the package using `apt`, I can run the following script (must be run as root user):

```python
import time
import uinput

def main():
    events = (
        uinput.REL_X,
        uinput.REL_Y,
        uinput.BTN_LEFT,
        uinput.BTN_RIGHT,
    )

    with uinput.Device(events) as device:
        for i in range(20):
            # syn=False to emit an "atomic" (5, 5) event.
            device.emit(uinput.REL_X, 5, syn=False)
            device.emit(uinput.REL_Y, 5)

            # Just for demonstration purposes: shows the motion. In real
            # application, this is of course unnecessary.
            time.sleep(0.01)

if __name__ == "__main__":
    main()
```

The `uinput` library simulate input events at the kernel level, so it works on both X11 and Wayland.
There is also an `evdev` package for python. That should also work (I did not try). It adds device listening as well, but we don't need that for this project.

## Communicating With The Extension
I can use [Native messaging](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Native_messaging) to interface with a local server that handles e.g. simulating mouse/keyboard events.

The [Zotero Connector](https://github.com/zotero/zotero-connectors) that I am familiar with (as a user) works differently: It expects the desktop client to run a server on a fixed port, and sends HTTP requests to it from a background script.
CORS prevents making the request directly from the content script, so we must send a message to a background script instead, and the background script can then make the request for us.

## Typescript Build
Rollup is giving me issues, so I'm trying out bun for a change:
```bash
bun build experiments/robot/extension/content/main.ts --outfile experiments/robot/extension/build/content.js
```
