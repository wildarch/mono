# Wraps the native sh_binary rule to provide a lint test automatically
def sh_binary(name = None, srcs = None, **kwargs):
    native.sh_binary(
        name = name,
        srcs = srcs,
        **kwargs
    )

    sh_lint_test(name = name + "_lint_test", deps = [":" + name])

def _sh_lint_test_impl(ctx):
    files_to_check = depset(transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps])
    paths = [f.path for f in files_to_check.to_list() if f.extension == "sh"]

    content = "{} {}".format(ctx.file._shellcheck.path, " ".join(paths))
    ctx.actions.write(
        output = ctx.outputs.executable,
        content = content,
    )

    runfiles = ctx.runfiles(files = [ctx.file._shellcheck], transitive_files = files_to_check)
    return [DefaultInfo(runfiles = runfiles)]

sh_lint_test = rule(
    implementation = _sh_lint_test_impl,
    attrs = {
        "deps": attr.label_list(providers = [DefaultInfo]),
        "_shellcheck": attr.label(allow_single_file = True, default = Label("@system-bin//:shellcheck")),
    },
    test = True,
)