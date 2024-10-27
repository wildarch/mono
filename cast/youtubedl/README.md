To download a youtube video with subs:

```
youtube-dl --write-auto-sub '<uri>'
```

Download it as mp4 with `yt-dlp`:

```
yt-dlp --write-auto-sub --format mp4 <uri>
```

To upload a directory containing processed video files (*Also uploads files in subdirectories*):

```
oci os object bulk-upload --bucket-name <bucket name> --src-dir <directory with videos>
```