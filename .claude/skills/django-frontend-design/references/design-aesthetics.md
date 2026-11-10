# Design Aesthetics for Django UIs

The goal: distinctive, production-grade interfaces that avoid generic "AI slop." The same bar as any world-class frontend work, applied inside Django templates.

## Design thinking (before writing markup)

Commit to a BOLD aesthetic direction:

- **Purpose**: What problem does this interface solve? Who uses it?
- **Tone**: Pick an extreme and execute it precisely — brutally minimal, maximalist, retro-futuristic, organic/natural, luxury/refined, playful/toy-like, editorial/magazine, brutalist/raw, art deco/geometric, soft/pastel, industrial/utilitarian. Use as inspiration; design one true to the chosen direction.
- **Constraints**: Framework, performance, accessibility, and the Django context (server-rendered, progressive enhancement via HTMX).
- **Differentiation**: What's the one thing someone will remember?

Bold maximalism and refined minimalism both work — the key is intentionality, not intensity.

## Guidelines

- **Typography**: Avoid generic fonts (Inter, Roboto, Arial, system defaults). Pick distinctive, characterful fonts; pair a display font with a refined body font. Load them in the base template `head` block (Google Fonts, Fontshare, or self-hosted via `{% static %}`).
- **Color & theme**: Commit to a cohesive palette. Use CSS variables, or Tailwind's `theme.extend.colors` / Bootstrap CSS variable overrides, for consistency. Dominant colors with sharp accents outperform timid, evenly distributed palettes.
- **Motion**: High-impact moments over scattered micro-interactions. One orchestrated page load with staggered reveals (`animation-delay`) delights more than dozens of tiny effects. CSS-first; layer HTMX swap transitions and Alpine where they fit.
- **Spatial composition**: Unexpected layouts. Asymmetry, overlap, diagonal flow, grid-breaking elements. Generous negative space OR controlled density.
- **Backgrounds & detail**: Atmosphere and depth over flat solid colors — gradient meshes, noise/grain textures, geometric patterns, layered transparency, dramatic shadows, decorative borders, custom cursors.

## Never

- Generic AI aesthetics: overused font families, cliched schemes (especially purple gradients on white), predictable layouts, cookie-cutter components.
- Converging on the same look across projects. Vary light/dark, fonts, and aesthetic each time. Don't default to the same "safe" display font repeatedly.

## Match complexity to vision

Maximalist designs need elaborate code with extensive animation and effects. Minimalist/refined designs need restraint, precision, and careful attention to spacing, typography, and subtle detail. Elegance comes from executing the chosen vision well — not from piling on effects.

## Django-specific application

- Define theme tokens once (Tailwind config or CSS variables in the base template) so every template inherits the aesthetic.
- Build the distinctive chrome (nav, footer, page transitions) into `base.html` so it carries across all pages for free.
- Keep HTMX-swapped partials visually consistent with the full page by sharing the same component markup.
- Distinctive fonts and the main stylesheet load through `{% static %}` / the `head` block — set this up once in the base template.
