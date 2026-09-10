/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'tactical-gray': '#1a1a24',
        'tactical-dark': '#0d0d14',
        'neon-cyan': '#00f3ff',
        'neon-red': '#ff2a2a',
      },
      fontFamily: {
        'mono': ['"JetBrains Mono"', 'monospace'],
        'sans': ['Inter', 'sans-serif'],
      },
      keyframes: {
        'zoom-out': {
          '0%': { transform: 'scale(1.15)' },
          '100%': { transform: 'scale(1)' },
        },
        'fade-in-up': {
          '0%': { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      },
      animation: {
        'slow-zoom': 'zoom-out 45s ease-out forwards',
        'fade-in-up': 'fade-in-up 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'fade-in': 'fade-in 1s ease-out forwards',
        'float': 'float 6s ease-in-out infinite',
      }
    },
  },
  plugins: [],
}
