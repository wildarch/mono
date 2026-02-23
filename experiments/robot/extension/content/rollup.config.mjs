import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';

const SRC_ROOT = 'experiments/robot/extension/content/';
const BUILD_DIR = 'experiments/robot/extension/build/'

const TS_OPTIONS = {
    compilerOptions: {
        lib: ['es2023', 'dom'],
        allowSyntheticDefaultImports: true
    },
};

export default {
    input: SRC_ROOT + 'main.ts',
    output: {
        file: BUILD_DIR + 'content.js',
        format: 'iife',
    },
    plugins: [resolve(), typescript(TS_OPTIONS)]
};
