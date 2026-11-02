import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// Dev server proxies /api to the FastAPI backend (default 127.0.0.1:8125) so
// the Vite dev server and the built app see the same API surface.
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8125",
    },
  },
});
