# Implementation Plan - Cyberpunk Theme

## Goal
Apply a "Cyberpunk" aesthetic to the dashboard, characterized by dark backgrounds, neon accents (cyan, purple, pink), glassmorphism, and futuristic UI elements.

## Current State
- The project uses React + Vite.
- Tailwind CSS classes are present in the JSX code (e.g., `text-cyan-400`, `p-6`), but **Tailwind CSS is NOT installed** in `package.json`.
- `index.css` contains some custom variables but is insufficient for the full design.

## Proposed Changes

### 1. Install & Configure Tailwind CSS
Since the code already uses Tailwind classes, we must install it to make them work.
- Install `tailwindcss`, `postcss`, `autoprefixer`.
- Initialize `tailwind.config.js` and `postcss.config.js`.
- Configure `tailwind.config.js` with the cyberpunk color palette.
- Add `@tailwind` directives to `index.css`.

### 2. Theming (Cyberpunk Aesthetics)
- **Colors:**
    - Background: Deep Black/Grey (`#050505`, `#0a0a0a`)
    - Primary: Neon Cyan (`#00f3ff`)
    - Secondary: Neon Purple (`#bc13fe`)
    - Accent: Neon Pink (`#ff0055`)
    - Success: Neon Green (`#0aff00`)
- **Effects:**
    - **Glassmorphism:** Translucent panels with blur and thin borders.
    - **Glow:** Box-shadows with colored spread.
    - **Grid Background:** A subtle retro-grid overlay.
    - **Scanlines:** Optional CRT scanline overlay.
- **Typography:**
    - Use a tech-inspired font (e.g., 'Orbitron' or 'Rajdhani') via Google Fonts.

### 3. Code Changes
- **`dashboard/src/index.css`**:
    - Import Google Fonts.
    - Define CSS variables for neon colors.
    - Add utility classes for `.glass-panel`, `.neon-text`, `.neon-border`.
    - Add background grid animation.
- **`dashboard/tailwind.config.js`**:
    - Extend theme with custom colors and fonts.
- **`dashboard/src/App.jsx`**:
    - Apply the new background and layout classes.
- **Components**:
    - Update charts (`recharts`) to use the new neon color palette.

## Verification Plan
1.  **Build Check:** Run `npm run build` to ensure Tailwind compiles correctly.
2.  **Visual Check:** Since I cannot see the screen, I will verify that:
    - `tailwind.config.js` exists and has the correct config.
    - `index.css` has the correct imports.
    - The server starts without errors.
