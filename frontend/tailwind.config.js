import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        page: "#0f172a",
        panel: "#1e293b",
        elevated: "#334155",
        accent: {
          primary: "#38bdf8",
          success: "#4ade80",
          danger: "#f87171",
          warning: "#fbbf24",
        },
        ink: {
          primary: "#f1f5f9",
          secondary: "#94a3b8",
        },
        line: "#334155",
      },
    },
  },
  plugins: [typography],
};
