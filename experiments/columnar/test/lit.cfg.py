from lit.llvm import llvm_config
import lit.formats

config.name = "Columnar"

config.test_format = lit.formats.ShTest(execute_external=False)

config.suffixes = ['.mlir', '.sql']

# Find common tools such as FileCheck
llvm_config.use_default_substitutions()

# The tools we want to use in lit test (inside RUN)
tools = [
    "mlir-opt",
    "translate",
]

# Where we look for the tools
tool_dirs = [
    config.llvm_tools_dir,
    config.mlir_tools_dir,
    config.columnar_tools_dir,
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.environment["FILECHECK_OPTS"] = "--enable-var-scope --allow-unused-prefixes=false"