# Tailwind CSS Setup in Django

Three ways to bring Tailwind into a Django project. Pick based on whether a Node build step is acceptable and whether the output is production-bound.

## Decision guide

| Approach | Build step | Node needed | Purging | Best for |
|---|---|---|---|---|
| Play CDN | none | no | no | prototypes, demos, admin tweaks, artifacts |
| django-tailwind | managed | yes | yes | teams wanting the "Django way" |
| Standalone CLI / npm | manual | CLI: no / npm: yes | yes | production, full control |

---

## 1. Play CDN (no build)

Drop this in the `<head>` of `base.html`. Works instantly, no toolchain.

```html
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    theme: {
      extend: {
        fontFamily: { display: ['"Fraunces"', 'serif'], body: ['"Inter Tight"', 'sans-serif'] },
        colors: { ink: '#1a1a1a', accent: '#e63946' },
      },
    },
  };
</script>
```

Limitations: no purge (large payload), no official plugins via npm, not for production. Good enough for quick UI work where you want results in one step.

---

## 2. django-tailwind package

Manages the Node/Tailwind build through a Django app and gives management commands.

```bash
pip install django-tailwind
# optional live reload:
pip install 'django-tailwind[reload]'
```

`settings.py`:

```python
INSTALLED_APPS = [
    # ...
    "tailwind",
    "theme",            # created next
    # "django_browser_reload",  # if using [reload]
]
TAILWIND_APP_NAME = "theme"
INTERNAL_IPS = ["127.0.0.1"]
```

Then:

```bash
python manage.py tailwind init      # creates the "theme" app
python manage.py tailwind install   # installs node deps
python manage.py tailwind start     # dev: watches & rebuilds
python manage.py tailwind build     # prod: minified output
```

In `base.html`:

```html
{% load static tailwind_tags %}
<head>
  {% tailwind_css %}
</head>
```

Requires Node.js installed on the dev machine. Output is handled through Django static files automatically.

---

## 3. Standalone CLI / npm (production-correct)

### Option A — Standalone CLI (no Node project)

Download the Tailwind standalone binary (no `package.json`, no `node_modules`). Create `static/src/input.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

`tailwind.config.js` at project root — point `content` at every template so unused classes are purged:

```js
module.exports = {
  content: [
    "./templates/**/*.html",
    "./**/templates/**/*.html",
  ],
  theme: {
    extend: {
      fontFamily: { display: ['Fraunces', 'serif'], body: ['"Inter Tight"', 'sans-serif'] },
    },
  },
  plugins: [],
};
```

Build:

```bash
# dev (watch)
./tailwindcss -i ./static/src/input.css -o ./static/css/output.css --watch
# prod (minified)
./tailwindcss -i ./static/src/input.css -o ./static/css/output.css --minify
```

### Option B — npm

```bash
npm init -y
npm install -D tailwindcss
npx tailwindcss init
```

Same `tailwind.config.js` content paths as above. Add scripts to `package.json`:

```json
{
  "scripts": {
    "dev": "tailwindcss -i ./static/src/input.css -o ./static/css/output.css --watch",
    "build": "tailwindcss -i ./static/src/input.css -o ./static/css/output.css --minify"
  }
}
```

### Wiring into Django (both options)

`settings.py`:

```python
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"   # for collectstatic in prod
```

`base.html`:

```html
{% load static %}
<head>
  <link rel="stylesheet" href="{% static 'css/output.css' %}">
</head>
```

Production: run the `--minify` build, then `python manage.py collectstatic`. Serve via WhiteNoise or your web server. Commit `output.css` is optional — usually it's a build artifact regenerated in CI.

### Tailwind v4 note

If using Tailwind v4, config moves into CSS (`@import "tailwindcss";` and `@theme { ... }`) and content detection is automatic. Confirm the installed major version before writing config, since v3 (`tailwind.config.js`) and v4 (CSS-based) differ. Check `npx tailwindcss --help` or the installed version.
