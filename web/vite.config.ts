import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies to the local MIMIR API. The UI never talks to a
// privileged endpoint directly from a remote origin (ADR 15).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8756", changeOrigin: false },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
