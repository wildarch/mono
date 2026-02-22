import typescript from '@rollup/plugin-typescript';

const SRC_ROOT = 'experiments/robot/extension/content/';
const BUILD_DIR = 'experiments/robot/extension/build/'

export default {
    input: SRC_ROOT + 'main.ts',
    output: {
        file: BUILD_DIR + 'content.js',
        format: 'iife'
    },
    plugins: [typescript()]
};
