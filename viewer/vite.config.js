import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Serve the repo's robot/ directory as static files so the URDF's
// package://hexapod_mkii_mock_assy/... URIs resolve without duplicating meshes.
export default defineConfig({
  plugins: [react()],
  publicDir: path.resolve(__dirname, '../robot'),
});
