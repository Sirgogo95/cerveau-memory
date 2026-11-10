# UX Design Discipline

This file covers the design thinking that happens *before* any markup is written. An expert UX designer owns the full problem of how people experience a product — from understanding users to validating that solutions work — while balancing user needs against business and technical realities. Skipping this and jumping straight to building produces pretty screens that solve the wrong problem.

## Table of contents

1. The expert mindset
2. Core responsibilities
3. Organizing an app: goals → flows → screens
4. Designing states, not just screens
5. Information architecture
6. Usability testing and validation
7. When to research vs. act
8. Accessibility and ethics baked in

---

## 1. The expert mindset

What separates an expert from a junior is not tooling — it's judgment:

- **Frame the right problem before solving it.** A request for "a settings page" might really be "users can't find how to change their plan." Restate the underlying goal before designing.
- **Tie everything to business goals.** Every flow should map to both a real user goal and a business need. If you can't name both, question whether the screen should exist.
- **Be comfortable with ambiguity.** Make reasonable assumptions explicit, design for the likely case, and note what you'd validate.
- **Strong judgment on when to research vs. act.** Don't over-research the obvious; don't act blindly on the unknown (see section 7).
- **Advocate for the user** while respecting engineering and product constraints.

When in doubt, state your assumptions about who the user is and what they're trying to accomplish, then design against those — and flag the riskiest assumption so the user can correct it.

---

## 2. Core responsibilities

An expert covers all of these, weighting them to the situation:

- **User research** — interviews, surveys, usability tests, field studies to understand needs, behaviors, and pain points. Synthesize into personas, journey maps, and insights rather than raw notes.
- **Information architecture** — structuring content and features so navigation is intuitive (sitemaps, navigation, taxonomies, hierarchies). See section 5.
- **Interaction design** — defining flows, states, transitions, and behaviors; mapping user journeys. See section 3 and the flow-diagrams reference.
- **Wireframing and prototyping** — low- and high-fidelity wireframes and interactive prototypes to test ideas before building them.
- **Usability testing and validation** — measuring whether designs work and refining based on evidence. See section 6.
- **Collaboration** — working with PMs, engineers, and stakeholders while keeping the user's interests central.
- **Design systems and standards** — consistent patterns, components, and accessibility standards (this connects directly to the Django implementation phase and the design-aesthetics reference).

---

## 3. Organizing an app: goals → flows → screens

The guiding principle: **work from user goals down to individual screens, never the reverse.** Designing screen-first produces a pile of pages no journey strings together.

1. **Start with goals and tasks, not screens.** List what users are trying to accomplish and the business outcome each serves. Screens come last.
2. **Map the primary user journeys.** Nail the core happy paths first — onboarding, the main task, account management — before edge cases.
3. **Build a clear information architecture.** Prefer shallow over deep; labels should match how users think, not internal jargon (section 5).
4. **Sequence each flow as steps.** Minimize steps. Ask for information only when it's actually needed. Front-load value and defer friction. One primary action per screen.
5. **Define entry and exit points.** Account for deep links, notifications, and users returning mid-task. Every screen should hold up as a possible *starting* point, not just a step in the middle.
6. **Design states, not just screens** (section 4).
7. **Handle branches and edge cases.** Map decision points and plan recovery paths so a user is never stranded.
8. **Maintain orientation and feedback.** Users should always know where they are, how far along they are, and what just happened.
9. **Prototype and test the flow.** Clickable prototypes tested early; watch for hesitation and drop-off.
10. **Iterate with data.** After launch, use analytics and funnels to find drop-off points and refine.

The key artifact for this work is a **flow diagram** — see `references/flow-diagrams.md` for how to produce one.

---

## 4. Designing states, not just screens

A "screen" is not one image — it's a set of states. For any meaningful screen, design for:

- **Empty** — first use, no data yet. This is an onboarding opportunity, not a blank void. Show what goes here and how to add it.
- **Loading** — skeletons or progress, not a frozen UI. Match perceived wait to actual wait.
- **Partial** — some data loaded, more coming, or a half-completed form.
- **Success** — confirm what happened and offer the obvious next action.
- **Error** — say what went wrong in plain language, why, and how to recover. Never a dead end.

In Django/HTMX terms these map cleanly to partial templates returned by views (empty-state partial, loading indicator via `hx-indicator`, error partial swapped into the target). Carry the state thinking from design straight into the implementation phase.

---

## 5. Information architecture

- **Shallow beats deep.** Fewer levels of nesting; more reachable in one or two clicks.
- **Label by user mental model.** Use the words users use. Test labels with a quick card sort if there's real ambiguity.
- **Group by task, not by org chart.** Don't expose your database tables or team structure as navigation.
- **Make the primary navigation reflect primary goals.** If three things matter, three things are prominent.

Produce a **sitemap** (hierarchy of sections/pages) alongside the flow diagram when structure is non-trivial.

---

## 6. Usability testing and validation

Designs are hypotheses until tested. Lightweight methods that fit most projects:

- **Hallway / 5-user tests.** Five users surface the majority of serious usability problems. Give them a real task, stay quiet, watch where they hesitate.
- **Watch for signals, not opinions.** Hesitation, backtracking, misclicks, and "wait, where am I?" matter more than "I like it."
- **Define success criteria up front.** Task completion rate, time on task, error rate, drop-off at each funnel step.
- **Post-launch:** instrument funnels and analytics; find the steps where people leave and refine those first.

When the user asks for validation help, propose a concrete test plan: the task, what to measure, how many users, and what would count as a pass.

---

## 7. When to research vs. act

A judgment call, not a rule:

- **Act directly** when the pattern is well-established (login, checkout, settings), the cost of being wrong is low, or you can ship and measure quickly.
- **Research first** when the problem is novel, the audience is unfamiliar, the cost of rework is high, or stakeholders disagree about what users actually need.

State which mode you're in and why. Don't pad a simple request with unnecessary research ceremony, and don't guess confidently on something genuinely unknown.

---

## 8. Accessibility and ethics baked in

Not a final-pass checklist — a design constraint from the start:

- **Accessibility:** sufficient color contrast (WCAG AA min), keyboard navigability, focus states, semantic HTML, labels on every input, alt text, respect for reduced-motion preferences. These carry directly into the markup in the build phase.
- **Ethics:** no dark patterns — no tricking users into purchases, hiding cancellation, manufacturing false urgency, or harvesting more data than the task needs. Front-load value and defer friction honestly; that's persuasion, not coercion.

Carry these into the implementation phase: semantic templates, labeled form fields, `prefers-reduced-motion` media queries, and contrast-checked palettes.
