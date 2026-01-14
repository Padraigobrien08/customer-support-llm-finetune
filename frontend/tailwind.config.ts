import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      colors: {
        background: "#0b1020",
        foreground: "#e6e9f2",
        panel: "#121a2b",
        border: "#202a44",
        accent: "#7aa2ff",
        accentSoft: "#243455",
        highlight: "#9ddcff"
      },
      boxShadow: {
        soft: "0 10px 30px rgba(0, 0, 0, 0.35)"
      }
    }
  },
  plugins: []
} satisfies Config;
