workspace(name = "com_github_wildarch_mono")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# System binaries
new_local_repository(
    name = "system-bin",
    build_file_content = """
exports_files([
    "rclone",
    "shellcheck",
    "openssl",
], visibility = ["//visibility:public"])
    """,
    path = "/usr/bin",
)

# Raspberry Pi image
http_file(
    name = "raspberry_pi_os_lite",
    sha256 = "34987327503fac1076e53f3584f95ca5f41a6a790943f1979262d58d62b04175",
    urls = ["https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2022-04-07/2022-04-04-raspios-bullseye-armhf-lite.img.xz"],
)

http_file(
    name = "zoom_deb",
    sha256 = "e2505be37fb44f7778435a50557cf8addcd89b8754302208b924a25887920470",
    urls = ["https://cdn.zoom.us/prod/5.11.0.3540/zoom_amd64.deb"],
)