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
        // Default: Flask serves media directly (no nginx needed).
        // For large collections, start nginx (see webapp/nginx.conf) and set:
        //   VITE_MEDIA_SERVER=http://127.0.0.1:8888 npm run dev
        target: process.env.VITE_MEDIA_SERVER || 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
