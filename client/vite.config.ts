import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  server: {
    port: 3000,
  },
  optimizeDeps: {
    // 👇 CRITICAL: Exclude both libraries from pre-bundling
    exclude: ['@xenova/transformers', 'onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
});
