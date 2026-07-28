/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Risk classes are colour-coded consistently with the CLI (ADR 13.2).
        risk: {
          r0: "#6b7280",
          r1: "#10b981",
          r2: "#f59e0b",
          r3: "#f97316",
          r4: "#ef4444",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
