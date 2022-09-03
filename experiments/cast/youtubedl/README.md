To download a youtube video with subs:

```
youtube-dl --write-auto-sub '<uri>'
```

To upload a directory containing processed video files (*Also uploads files in subdirectories*):

```
oci os object bulk-upload --bucket-name <bucket name> --src-dir <directory with videos>
```