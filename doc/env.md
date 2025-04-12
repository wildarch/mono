# Moving away from Bazel
Bazel has been nice to use for my monorepo:
- Elegant syntax for declaring build targets
- A single tool to build/run/test the whole repo
- Hermetic, reliable builds

I have also found Bazel annoying or limiting to use at times:
- Not all languages support Bazel. This is sometimes annoying when I want to experiment more niche programming languages
- Package management. Mainly for C++, most external projects assume CMake, and are difficult to integrate. For example, I cannot use Apache Arrow because it lacks Bazel build instructions.
- Tool support. Editor plugins frequently misbehave in Bazel workspaces because they expect some other, language-specific build tool.

I want to try and use a devcontainer instead for a more flexible solution.
This means foregoing a central tool for building and testing parts of the repo, but in return I get more control over how I build and run my code.
The devcontainer should keep things relatively reliable/portable.

## Devcontainer
Steps:
- [x] Build cast backend
- [ ] Build columnar
- [ ] Tweak ansible deployment scripts
- [ ] Update CI
