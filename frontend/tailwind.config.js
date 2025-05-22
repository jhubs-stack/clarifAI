/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      animation: {
        'chat-bounce': 'bounce 2s infinite',
        'sparkle': 'sparkle 1s infinite ease-in-out',
        'fade-in': 'fadeIn 0.3s ease-out',
      },
      keyframes: {
        sparkle: {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.3)', opacity: '0.7' },
        },
        fadeIn: {
          from: { opacity: 0, transform: 'translateY(10px)' },
          to: { opacity: 1, transform: 'translateY(0)' },
        }
      },
    },
  },
  plugins: [],
}