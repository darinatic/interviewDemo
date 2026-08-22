import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Relative base so the built page works from any subpath - GitHub Pages project
  // sites, a Netlify preview URL, or opened straight off disk. An absolute base is
  // the usual reason a static demo 404s its own assets when someone shares it.
  base: "./",
  build: { outDir: "dist", sourcemap: false },
});
