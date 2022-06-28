workspace(name = "com_github_wildarch_mono")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//package/zoom:zoom.bzl", "ZOOM_SHA256", "ZOOM_URL")

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
    sha256 = ZOOM_SHA256,
    urls = [ZOOM_URL],
)

# Kotlin
rules_kotlin_version = "1.6.0"

rules_kotlin_sha = "a57591404423a52bd6b18ebba7979e8cd2243534736c5c94d35c89718ea38f94"

http_archive(
    name = "io_bazel_rules_kotlin",
    sha256 = rules_kotlin_sha,
    urls = ["https://github.com/bazelbuild/rules_kotlin/releases/download/v%s/rules_kotlin_release.tgz" % rules_kotlin_version],
)

load("@io_bazel_rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()  # if you want the default. Otherwise see custom kotlinc distribution below

load("@io_bazel_rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()  # to use the default toolchain, otherwise see toolchains below

# ANTLR
http_archive(
    name = "rules_antlr",
    sha256 = "26e6a83c665cf6c1093b628b3a749071322f0f70305d12ede30909695ed85591",
    strip_prefix = "rules_antlr-0.5.0",
    urls = ["https://github.com/marcohu/rules_antlr/archive/0.5.0.tar.gz"],
)

load("@rules_antlr//antlr:repositories.bzl", "rules_antlr_dependencies")

rules_antlr_dependencies("4.8")
