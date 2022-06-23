workspace(name = "com_github_wildarch_mono")

# System binaries
new_local_repository(
    name = "system-bin",
    build_file_content = """
exports_files([
    "rclone",
    "shellcheck",
], visibility = ["//visibility:public"])
    """,
    path = "/usr/bin",
)