import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Allow external connections
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000', // Direct to backend, bypass nginx
        changeOrigin: true,
        secure: false,
      },
       '/media': {
        target: 'http://127.0.0.1:8888',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
