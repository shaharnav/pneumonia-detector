/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        "clinical-blue": "#1d4ed8",
        "clinical-blue-soft": "#dbeafe",
      },
    },
  },
  plugins: [],
};

