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
    sha256 = "5df1850573c5e1418f70285c96deea2cfa87105cca976262f023c49b31cdd52b",
    urls = ["https://downloads.raspberrypi.com/raspios_lite_armhf/images/raspios_lite_armhf-2023-12-11/2023-12-11-raspios-bookworm-armhf-lite.img.xz"],
)

# ANTLR
http_archive(
    name = "rules_antlr",
    sha256 = "26e6a83c665cf6c1093b628b3a749071322f0f70305d12ede30909695ed85591",
    strip_prefix = "rules_antlr-0.5.0",
    urls = ["https://github.com/marcohu/rules_antlr/archive/0.5.0.tar.gz"],
)

load("@rules_antlr//antlr:repositories.bzl", "rules_antlr_dependencies")

rules_antlr_dependencies("4.8")

# rust
http_archive(
    name = "rules_rust",
    sha256 = "dd79bd4e2e2adabae738c5e93c36d351cf18071ff2acf6590190acf4138984f6",
    urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.14.0/rules_rust-v0.14.0.tar.gz"],
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

rust_register_toolchains()

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")

crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crates_repository")

crates_repository(
    name = "crate_index",
    cargo_lockfile = "//:Cargo.lock",
    lockfile = "//:Cargo.Bazel.lock",
    manifests = [
        "//:Cargo.toml",
        "//:experiments/dblp/dblp-rs/Cargo.toml",
    ],
)

load("@crate_index//:defs.bzl", "crate_repositories")

crate_repositories()
