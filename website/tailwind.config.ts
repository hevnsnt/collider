import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Base backgrounds
        background: {
          DEFAULT: "#0a0a0a",
          secondary: "#111111",
          tertiary: "#1a1a1a",
          elevated: "#222222",
        },
        // Text colors
        foreground: {
          DEFAULT: "#e0e0e0",
          secondary: "#888888",
          muted: "#555555",
        },
        // Primary accent - Electric Cyan
        cyan: {
          DEFAULT: "#00ffff",
          dim: "#00cccc",
          glow: "rgba(0, 255, 255, 0.3)",
          50: "#ecfeff",
          100: "#cffafe",
          200: "#a5f3fc",
          300: "#67e8f9",
          400: "#22d3ee",
          500: "#00ffff",
          600: "#00cccc",
          700: "#0e7490",
          800: "#155e75",
          900: "#164e63",
        },
        // Secondary accent - Amber
        amber: {
          DEFAULT: "#ffb000",
          dim: "#cc8800",
          glow: "rgba(255, 176, 0, 0.3)",
          50: "#fffbeb",
          100: "#fef3c7",
          200: "#fde68a",
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#ffb000",
          600: "#cc8800",
          700: "#b45309",
          800: "#92400e",
          900: "#78350f",
        },
        // Success - Green
        success: {
          DEFAULT: "#00ff88",
          dim: "#00cc66",
        },
        // Error - Red
        error: {
          DEFAULT: "#ff3366",
          dim: "#cc2952",
        },
        // Borders
        border: {
          DEFAULT: "#333333",
          hover: "#444444",
          accent: "#00ffff",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "SF Mono", "Fira Code", "monospace"],
        sans: ["Geist", "Inter", "system-ui", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.75rem" }],
      },
      animation: {
        "matrix-rain": "matrixRain 20s linear infinite",
        "glow-pulse": "glowPulse 2s ease-in-out infinite",
        "scan-line": "scanLine 8s linear infinite",
        "glitch": "glitch 0.3s ease-in-out",
        "typewriter": "typewriter 2s steps(40) forwards",
        "blink": "blink 1s step-end infinite",
        "float": "float 6s ease-in-out infinite",
        "data-stream": "dataStream 3s linear infinite",
        "counter": "counter 2s ease-out forwards",
        "fade-in": "fadeIn 0.5s ease-out forwards",
        "fade-in-up": "fadeInUp 0.6s ease-out forwards",
        "slide-in-left": "slideInLeft 0.5s ease-out forwards",
        "slide-in-right": "slideInRight 0.5s ease-out forwards",
        "scale-in": "scaleIn 0.3s ease-out forwards",
      },
      keyframes: {
        matrixRain: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        glowPulse: {
          "0%, 100%": { opacity: "1", boxShadow: "0 0 20px rgba(0, 255, 255, 0.5)" },
          "50%": { opacity: "0.7", boxShadow: "0 0 40px rgba(0, 255, 255, 0.8)" },
        },
        scanLine: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        glitch: {
          "0%": { transform: "translate(0)" },
          "20%": { transform: "translate(-2px, 2px)" },
          "40%": { transform: "translate(-2px, -2px)" },
          "60%": { transform: "translate(2px, 2px)" },
          "80%": { transform: "translate(2px, -2px)" },
          "100%": { transform: "translate(0)" },
        },
        typewriter: {
          "from": { width: "0" },
          "to": { width: "100%" },
        },
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        dataStream: {
          "0%": { backgroundPosition: "0% 0%" },
          "100%": { backgroundPosition: "0% 100%" },
        },
        counter: {
          "from": { opacity: "0", transform: "translateY(10px)" },
          "to": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "from": { opacity: "0" },
          "to": { opacity: "1" },
        },
        fadeInUp: {
          "from": { opacity: "0", transform: "translateY(20px)" },
          "to": { opacity: "1", transform: "translateY(0)" },
        },
        slideInLeft: {
          "from": { opacity: "0", transform: "translateX(-20px)" },
          "to": { opacity: "1", transform: "translateX(0)" },
        },
        slideInRight: {
          "from": { opacity: "0", transform: "translateX(20px)" },
          "to": { opacity: "1", transform: "translateX(0)" },
        },
        scaleIn: {
          "from": { opacity: "0", transform: "scale(0.95)" },
          "to": { opacity: "1", transform: "scale(1)" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-hero": "linear-gradient(135deg, #0a0a0a 0%, #111827 50%, #0a0a0a 100%)",
        "gradient-card": "linear-gradient(145deg, #1a1a1a 0%, #111111 100%)",
        "grid-pattern": "linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px)",
        "noise": "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.05'/%3E%3C/svg%3E\")",
      },
      backgroundSize: {
        "grid": "50px 50px",
      },
      boxShadow: {
        "glow-cyan": "0 0 20px rgba(0, 255, 255, 0.3), 0 0 40px rgba(0, 255, 255, 0.1)",
        "glow-amber": "0 0 20px rgba(255, 176, 0, 0.3), 0 0 40px rgba(255, 176, 0, 0.1)",
        "glow-green": "0 0 20px rgba(0, 255, 136, 0.3), 0 0 40px rgba(0, 255, 136, 0.1)",
        "inner-glow": "inset 0 0 20px rgba(0, 255, 255, 0.1)",
        "card": "0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3)",
        "card-hover": "0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.4), 0 0 30px rgba(0, 255, 255, 0.1)",
      },
      borderRadius: {
        "none": "0",
        "sm": "0.125rem",
        "md": "0.25rem",
        "lg": "0.5rem",
      },
      transitionDuration: {
        "400": "400ms",
      },
      spacing: {
        "18": "4.5rem",
        "88": "22rem",
        "128": "32rem",
      },
    },
  },
  plugins: [],
};

export default config;
