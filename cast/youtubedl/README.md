To download a youtube video with subs:

```
youtube-dl --write-auto-sub '<uri>'
```

Download it as mp4 with `yt-dlp`:

```
yt-dlp --write-auto-sub -t mp4 <uri>
```

NOTE: You must have the `ffmpeg` binary in your path to do the final mp4 format fixup.

To upload a directory containing processed video files (*Also uploads files in subdirectories*):

```
oci os object bulk-upload --bucket-name <bucket name> --src-dir <directory with videos>
```