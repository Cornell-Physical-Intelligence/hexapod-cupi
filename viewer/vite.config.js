import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Serve the approved robot assets from their canonical directory.
export default defineConfig({
  plugins: [react()],
  publicDir: path.resolve(__dirname, '../robot'),
});
