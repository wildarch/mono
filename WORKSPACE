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
rules_kotlin_version = "1.7.0-RC-3"

rules_kotlin_sha = "f033fa36f51073eae224f18428d9493966e67c27387728b6be2ebbdae43f140e"

http_archive(
    name = "io_bazel_rules_kotlin",
    sha256 = rules_kotlin_sha,
    urls = ["https://github.com/bazelbuild/rules_kotlin/releases/download/v%s/rules_kotlin_release.tgz" % rules_kotlin_version],
)

load("@io_bazel_rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()  # if you want the default. Otherwise see custom kotlinc distribution below

load("@io_bazel_rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()

# ANTLR
http_archive(
    name = "rules_antlr",
    sha256 = "26e6a83c665cf6c1093b628b3a749071322f0f70305d12ede30909695ed85591",
    strip_prefix = "rules_antlr-0.5.0",
    urls = ["https://github.com/marcohu/rules_antlr/archive/0.5.0.tar.gz"],
)

load("@rules_antlr//antlr:repositories.bzl", "rules_antlr_dependencies")

rules_antlr_dependencies("4.8")

# Maven
RULES_JVM_EXTERNAL_TAG = "4.2"

RULES_JVM_EXTERNAL_SHA = "cd1a77b7b02e8e008439ca76fd34f5b07aecb8c752961f9640dea15e9e5ba1ca"

http_archive(
    name = "rules_jvm_external",
    sha256 = RULES_JVM_EXTERNAL_SHA,
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:repositories.bzl", "rules_jvm_external_deps")

rules_jvm_external_deps()

load("@rules_jvm_external//:setup.bzl", "rules_jvm_external_setup")

rules_jvm_external_setup()

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
        "junit:junit:4.12",
        "com.google.truth:truth:1.1.3",
        "org.commonmark:commonmark:0.18.1",
        # Ktor
        "io.ktor:ktor-server-core-jvm:2.1.0",
        "io.ktor:ktor-server-netty-jvm:2.1.0",
        "io.ktor:ktor-server-html-builder-jvm:2.1.0",
        "io.ktor:ktor-utils-jvm:2.1.0",
        "io.ktor:ktor-http:2.1.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.6.4",
        "org.jetbrains.kotlinx:kotlinx-html-jvm:0.8.0",
        # OCI SDK
        "com.oracle.oci.sdk:oci-java-sdk-core:2.40.0",
        "com.oracle.oci.sdk:oci-java-sdk-objectstorage:2.40.0",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
)

# rules_pkg
http_archive(
    name = "rules_pkg",
    sha256 = "8a298e832762eda1830597d64fe7db58178aa84cd5926d76d5b744d6558941c2",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.7.0/rules_pkg-0.7.0.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.0/rules_pkg-0.7.0.tar.gz",
    ],
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

# golang
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "099a9fb96a376ccbbb7d291ed4ecbdfd42f6bc822ab77ae6f1b5cb9e914e94fa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.19.1")

http_archive(
    name = "bazel_gazelle",
    sha256 = "efbbba6ac1a4fd342d5122cbdfdb82aeb2cf2862e35022c752eaddffada7c3f3",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.27.0/bazel-gazelle-v0.27.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.27.0/bazel-gazelle-v0.27.0.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")
load("//:deps.bzl", "go_dependencies")

# gazelle:repository_macro deps.bzl%go_dependencies
go_dependencies()

gazelle_dependencies()

# rust
http_archive(
    name = "rules_rust",
    sha256 = "dd79bd4e2e2adabae738c5e93c36d351cf18071ff2acf6590190acf4138984f6",
    urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.14.0/rules_rust-v0.14.0.tar.gz"],
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

rust_register_toolchains()

# xmlwrapp
http_archive(
    name = "xmlwrapp",
    build_file_content = """
cc_library(
    name = "xmlwrapp",
    srcs = glob([
        "src/**/*.cxx",
        "src/**/*.h",
    ]),
    hdrs = glob([
        "include/**/*.h",
    ]),
    copts = ["-I/usr/include/libxml2"],
    linkopts = ["-lxml2"],
    strip_include_prefix = "include/",
    visibility = ["//visibility:public"],
)
    """,
    sha256 = "96c6229ce246edf9e48a8b566addd970a19eb9d4b51688afd104077e72679f75",
    strip_prefix = "xmlwrapp-0.9.1",
    urls = ["https://github.com/vslavik/xmlwrapp/releases/download/v0.9.1/xmlwrapp-0.9.1.tar.gz"],
)

# Boost
BOOST_COMMIT = "f1065639e6f33741abe2a6a78fa79dd1a07bbf5d"

BOOST_SHA256 = "6ded3e8c064054c92b79aeadde2d78821c889598e634c595133da0ea8f0f0b85"

http_archive(
    name = "com_github_nelhage_rules_boost",
    sha256 = BOOST_SHA256,
    strip_prefix = "rules_boost-%s" % BOOST_COMMIT,

    # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/nelhage/rules_boost/archive/%s.tar.gz" % BOOST_COMMIT,
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()
