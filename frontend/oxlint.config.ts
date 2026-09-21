import { defineConfig } from "oxlint";

export default defineConfig({
  plugins: ["typescript", "react", "oxc"],

  env: {
    browser: true,
  },

  ignorePatterns: [
    "dist/**",
    "coverage/**",
    "node_modules/**",
  ],

  options: {
    typeAware: true,
    typeCheck: true,
  },

  rules: {
    "eslint/no-unused-vars": "error",
    "typescript/no-explicit-any": "error",

    "react/hooks": "error",
    "react/exhaustive-deps": "error",

    "react/only-export-components": [
      "warn",
      {
        allowConstantExport: true,
      },
    ],
  },
});