import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import copy from 'rollup-plugin-copy'

export default defineConfig({
  plugins: [
    react(),
    copy({
      targets: [
        // Copy the tzas folder to the root of the dist directory
        //         { src: 'node_modules/denoiser/tzas', dest: 'dist' }
        // because of yarn the local path is wrong and we need to go to the project root

        { src: '../../node_modules/denoiser/tzas', dest: 'dist' }
      ],
      hook: 'writeBundle' // Ensures copying after the bundle is written
    })
  ],
})