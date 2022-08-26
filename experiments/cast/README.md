# Playing content to my Living Room TV
I do not watch too much TV, but when I feel like it, here is what I want:
- Little or no Ads. YouTube plays so many these days and it drives me nuts.
- Select and control playback with my phone
- Instantly play my favourite content
- Show subtitles (if available)

I regularly watch videos from these services:
- YouTube. Notably I am slowly making my way through all seasons of [Taskmaster](https://www.youtube.com/c/Taskmaster).
- NPO (Dutch public broadcaster)
- Netflix. Very sporadic, but I do have an account.

All of these have Chromecast integration, and I have happen to have a Chromecast plugged into my TV already, so that covers most of my needs.

The one problem with this setup is that YouTube shows way too many Ads. 
Now for the occasional video I could live with it, but it is annoying when watching Taskmaster.
It seems Chromecasts have a bug that disables subtitles after an ad break, so after every commercial I have to toggle them off and on again.

So, to address my main concern a minimum viable solution is one that allows me to play Taskmaster episodes to my TV with subtitles and without Ads.

# Current solution
I have a Raspberry Pi 4 in my home network that serves as a media downloader, converter and streamer.

A shell script downloads taskmaster episodes and recodes them to bake in subtitles (also downloaded from YouTube).
I run this script manually once I finish the current season.

A small go binary runs a web server that allows me to select one of the downloaded files on my phone, and instructs VLC to stream it to the chromecast.
There is currently no way to seek or pause the video, just a stop button.
Seek controls could be added with reasonable effort, though they would make the frontend much more involved, but in reality I haven't had a need for them so far.

## Pros
Running it locally makes the security aspect much easier to manage.

I do not have to protect the service against unauthorized users, and exposing the downloaded videos publicly would probably be illegal.

## Cons
My main annoyance is that the Pi is not really used for anything else. 
When I want to watch something I have to go turn it on, and I have to turn it off again.

# Hosted solution
Hosting media on a publicly accessible server makes the security aspect more difficult, but it does have some distinct advantages.
First of all, I could run it on a VPS that I already have, so it reduces maintainance (and energy consumption, slightly).
Most interestingly, this would effectively turn it into a world-wide accessible media library, and I could stream content to a chromecast no matter where I go, even on holiday.

While the current solution works well enough, I am curious what would be needed to upgrade it to the hosted solution.

## High-level Design
This design requires a backend to authenticate requests and serve media files, as well as a frontend that can shown videos and cast them to nearby devices using the Google Cast SDK.

### Authentication
This service has only one user, so the simplest way to do this is to have a password in a configuration file, and have the user go through HTTP basic auth when they visit the service.
For subsequent requests, they can authenticate with a cookie. 
When the chromecast opens a video file, it could specify the password (or a token) as a request parameter, assuming we don't have a way to pass coookies or anything to it.

### Video storage
Initially we can just store them on disk on the VPS, and have our web server stream them.
Eventually we could look into making the whole thing more stateless by using object storage (does that allow streaming? I'm not sure, haven't used it before).

### Casting
A decent tutorial on making videos on a website cast-able is [here](https://developers.google.com/cast/codelabs/cast-videos-chrome#0).
Casting requires a receiver application ID, but its fine to just use DEFAULT_MEDIA_RECEIVER_APP_ID.

Unfortunately there isn't a well-supported open source HTML5 video player library that supports casting, so we'd have to roll our own (or steal the one from the code sample).