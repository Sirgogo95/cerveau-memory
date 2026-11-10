---
name: django-frontend-design
description: Design and build distinctive, production-grade Django frontends — covering both the UX design discipline (user research, information architecture, user flows, screen states, usability testing) and the implementation (Tailwind CSS or Bootstrap, templates, forms, HTMX). Use whenever the user is working on a Django app and wants to plan how users move through it, structure its screens, map a user flow or journey, decide what pages are needed, OR create/style templates, pages, components, dashboards, forms, or any web UI. Triggers include "design the UX for my Django app," "how should this app flow," "map the user journey," "what screens do I need," "sketch a user flow diagram," "style my Django app," "build a Django template," "add Tailwind," "make this page look better," styling forms, template inheritance, or HTMX. Use it even when the user mentions Django plus a design, flow, or UI task without naming the framework, and even when they ask only for the planning/UX part with no code yet.
---

This skill guides the full arc of Django frontend work: first designing how people experience the product (UX), then building it (implementation). An expert owns both — a beautiful screen that solves the wrong problem is a failure, and a well-mapped flow with generic markup is a missed opportunity. Do the design thinking first, then implement it correctly and distinctively.

Tailwind CSS is the default for implementation. Use Bootstrap only when the user asks for it, the project already uses it, or a no-build CDN component library is explicitly preferred.

## Two phases

**Phase 1 — UX design** (do this first, even briefly): understand the user and goal, structure the app, map the flows and states. Skip or compress only when the task is purely cosmetic (e.g., "restyle this existing button").

**Phase 2 — Implementation**: build the templates, styling, forms, and interactivity that realize the design, the Django way.

Judge how much of Phase 1 a request needs. "Make this page look better" leans implementation. "Design the onboarding for my app" is mostly Phase 1 and produces a flow diagram before any code. Most substantial features touch both.

---

## Phase 1 — UX design

Read `references/ux-design.md` for the full discipline. The essentials:

1. **Frame the right problem.** Restate the underlying user goal and the business need before designing anything. If you can't name both, question whether the screen should exist.
2. **Work goals → flows → screens, never screen-first.** List user goals and tasks, map the primary journeys (happy paths first), then derive screens.
3. **Build a shallow, intuitive information architecture.** Labels match how users think; group by task, not org chart.
4. **Sequence each flow as minimal steps.** Ask for info only when needed, front-load value, defer friction, one primary action per screen. Account for entry/exit points (deep links, notifications, returning mid-task).
5. **Design states, not just screens.** Every meaningful screen needs empty, loading, partial, success, and error states — never a dead end. These map directly onto Django/HTMX partials in Phase 2.
6. **Validate.** Propose lightweight usability testing (5-user tests, watch for hesitation/drop-off) and post-launch funnel analysis when validation matters. See the research-vs-act guidance.
7. **Bake in accessibility and ethics from the start** — contrast, keyboard nav, semantic HTML, labeled inputs, reduced-motion; no dark patterns. These carry into the markup.

**Produce a flow diagram** as the key artifact whenever a feature spans more than a couple of screens or has any branching. Default to rendering a **visual diagram** (nodes = screens/states, labeled arrows = actions, diamonds = decision points, distinct styling for error/recovery paths). Use Mermaid when the user wants something pasteable into docs. Full guidance and node conventions in `references/flow-diagrams.md`. The diagram becomes the spec Phase 2 builds against.

When information architecture is non-trivial, also produce a **sitemap**.

---

## Phase 2 — Implementation

### Workflow

1. **Identify the setup.** Inspect the project (or ask): fresh or existing? Tailwind/Bootstrap already configured? Template structure (`templates/` layout, base template, app dirs)? Forms, HTMX, or a CSS build already present?
2. **Pick the integration approach** for Tailwind (see below) based on whether a Node build step is acceptable.
3. **Commit to an aesthetic direction** (see "Design quality") before writing markup.
4. **Build the templates and styling** — realizing the Phase 1 flow — respecting Django conventions (template inheritance, `{% static %}`, `{% csrf_token %}`, form rendering).
5. **Wire up interactivity** with HTMX where dynamic behavior is needed; implement the empty/loading/error states from Phase 1 as partials.
6. **Verify** static files resolve and the build (if any) produces output.

### Tailwind setup approaches

Pick ONE. Read `references/tailwind-setup.md` for full, copy-paste-ready instructions for each.

- **CDN (Play CDN)** — Fastest. No build step, no Node. One `<script>` in the base template. Best for prototypes, demos, admin tweaks, and artifacts. Not for production (no purging, larger payload). Use when the user wants something working immediately or has no Node toolchain.
- **django-tailwind package** — Integrates Tailwind through a Django app, manages the Node build inside the project, gives `tailwind start`/`tailwind build` commands, plays nicely with `{% static %}`. Best when the team wants Tailwind "the Django way."
- **Standalone CLI / npm build** — Most control and most production-correct. Compile `input.css` → `static/css/output.css` with content paths pointing at templates so unused classes are purged. Best for production apps wanting full config.

Default: **standalone CLI** for production-bound work, **CDN** for quick work. Confirm if unsure.

For Bootstrap, read `references/bootstrap.md` — CDN, `django-bootstrap5`, and form rendering with `crispy-forms`/`crispy-bootstrap5`.

### Django integration essentials

These make Django frontend work *correct*, not just pretty. Detailed patterns in `references/django-patterns.md`; the essentials:

- **Template inheritance**: One `base.html` with named `{% block %}` regions (`title`, `head`, `content`, `extra_js`). Children `{% extends "base.html" %}` and fill blocks. Shared chrome (nav, footer, flash messages) lives in the base.
- **Components**: Prefer `{% include %}` with context for reusable partials, or the `{% component %}` tag from `django-cotton` / `django-components` for prop-driven components. Keep partials in `templates/components/` or `partials/`, often `_`-prefixed.
- **Static files**: Always load via `{% load static %}` and `{% static 'css/output.css' %}` — never hardcode. Configure `STATIC_URL`, `STATICFILES_DIRS`, and (prod) `STATIC_ROOT` + `collectstatic`.
- **Forms**: Django renders fields without utility classes by default. Style via widget `attrs`, a form-rendering library, or per-field template rendering. See `references/django-patterns.md` for the three approaches.
- **CSRF**: Every POST form needs `{% csrf_token %}`. HTMX POSTs need the CSRF token wired in (header or hidden input).

### Interactivity with HTMX

HTMX is the recommended way to add dynamic behavior without a SPA. Read `references/htmx.md` for: including HTMX, CSRF setup, returning partial templates from views, swap strategies, and common patterns (inline edit, infinite scroll, active search, modals, form validation). Keep views returning small partials targeting a specific element via `hx-target`. This is also how Phase 1's empty/loading/error states get implemented.

### Design quality

Apply the bar of world-class frontend design. Before writing markup, commit to a BOLD, intentional aesthetic direction rather than generic output. Read `references/design-aesthetics.md`. Core rules:

- **Typography**: Avoid generic fonts (Inter, Roboto, Arial, system defaults). Pair a distinctive display font with a refined body font, loaded via the base template `head` block.
- **Color & theme**: Cohesive palette via CSS variables (or Tailwind theme config). Dominant colors with sharp accents beat timid, evenly distributed palettes. Vary between light and dark.
- **Motion**: High-impact moments over scattered micro-interactions. Respect `prefers-reduced-motion`. CSS-first; HTMX transitions where they fit.
- **Composition**: Unexpected layouts, asymmetry, overlap, grid-breaking elements, intentional negative space or controlled density.
- **Backgrounds & detail**: Atmosphere and depth over flat solids — gradient meshes, noise/grain, geometric patterns, layered transparency, dramatic shadows.

NEVER ship generic AI aesthetics: overused fonts, purple-gradient-on-white clichés, predictable layouts, cookie-cutter components. Match implementation complexity to the vision. Every design should feel made for *this* app's context.

### Output expectations

- Produce real, working Django templates and config — not pseudo-code.
- Show where each file goes in the project tree (`templates/`, `static/`, app dirs, `settings.py`).
- When introducing a build step, give exact commands and the relevant `settings.py` / config diffs.
- When styling forms, show the view, the form, and the template together.
- Note any `pip install` / `npm install` dependencies up front.

---

## Reference files

- `references/ux-design.md` — Full UX discipline: mindset, responsibilities, goals→flows→screens, states, IA, usability testing, research-vs-act, accessibility/ethics.
- `references/flow-diagrams.md` — How to produce user/task flow diagrams (visual or Mermaid), node conventions, and how the diagram maps onto Django templates/views.
- `references/tailwind-setup.md` — CDN, django-tailwind, and standalone CLI/npm setups with full config.
- `references/bootstrap.md` — Bootstrap CDN, django-bootstrap5, crispy-forms.
- `references/django-patterns.md` — Template inheritance, components, static files, form styling (three approaches), messages framework.
- `references/htmx.md` — HTMX include, CSRF wiring, partial views, swap patterns.
- `references/design-aesthetics.md` — Full aesthetic guidance for distinctive, non-generic UI.
