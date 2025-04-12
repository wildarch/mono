set (LLVM_EXTERNAL_LIT "/usr/local/bin/lit")
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check "Run integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS columnar-opt translate
)
