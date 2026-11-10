# Django Frontend Patterns

Correct, idiomatic Django integration for templates, components, static files, and forms.

## Template inheritance

One base template with named blocks; children extend and fill them.

`templates/base.html`:

```html
{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}My App{% endblock %}</title>
  <link rel="stylesheet" href="{% static 'css/output.css' %}">
  {% block head %}{% endblock %}
</head>
<body class="font-body bg-ink text-stone-100">
  {% include "partials/_nav.html" %}
  {% include "partials/_messages.html" %}
  <main>
    {% block content %}{% endblock %}
  </main>
  {% include "partials/_footer.html" %}
  <script src="https://unpkg.com/htmx.org@2"></script>
  {% block extra_js %}{% endblock %}
</body>
</html>
```

Child template:

```html
{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
  <section class="mx-auto max-w-5xl px-6 py-16">...</section>
{% endblock %}
```

## Components / partials

**Simple reuse** — `{% include %}` with context:

```html
{% include "components/_card.html" with title="Revenue" value="$48k" trend="up" %}
```

`templates/components/_card.html`:

```html
<article class="rounded-2xl border border-stone-800 bg-stone-900/60 p-6">
  <h3 class="text-sm uppercase tracking-widest text-stone-400">{{ title }}</h3>
  <p class="mt-2 font-display text-3xl">{{ value }}</p>
</article>
```

**Prop-driven components** — for anything more than trivial reuse, suggest `django-cotton` (HTML-syntax components, `<c-card title="...">`) or `django-components`. Mention these when the user has many repeated UI elements; don't force them for one-offs.

## Static files

- `{% load static %}` at the top of any template using assets.
- Reference assets with `{% static 'path' %}`, never hardcoded URLs.
- Dev: `STATICFILES_DIRS` points at your source `static/` dir.
- Prod: `STATIC_ROOT` + `python manage.py collectstatic`, served by WhiteNoise or the web server.

```python
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
# WhiteNoise (prod):
# STORAGES = {"staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"}}
```

## Styling Django forms

Django renders fields without utility classes by default. Three approaches:

### A. Widget attrs (no dependencies)

```python
class ContactForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            "class": "w-full rounded-lg border border-stone-700 bg-stone-900 px-4 py-2.5 "
                     "focus:border-accent focus:ring-2 focus:ring-accent/40",
            "placeholder": "you@example.com",
        })
    )
```

Then render `{{ form.email }}` and it carries the classes. Clean, but couples styling to Python.

### B. django-widget-tweaks (recommended for Tailwind)

```bash
pip install django-widget-tweaks
```

```html
{% load widget_tweaks %}
<label class="block text-sm text-stone-400">{{ form.email.label }}</label>
{% render_field form.email class="w-full rounded-lg border border-stone-700 bg-stone-900 px-4 py-2.5" placeholder="you@example.com" %}
{% if form.email.errors %}
  <p class="mt-1 text-sm text-accent">{{ form.email.errors.0 }}</p>
{% endif %}
```

Keeps styling in the template where it belongs. Best default for Tailwind projects.

### C. Manual field markup

Iterate fields and build the markup by hand for full control. Most verbose; use when layouts are bespoke.

### Full form template

```html
<form method="post" class="space-y-5">
  {% csrf_token %}
  {% for field in form %}
    <div>
      <label class="block text-sm text-stone-400" for="{{ field.id_for_label }}">{{ field.label }}</label>
      {% render_field field class="w-full rounded-lg border border-stone-700 bg-stone-900 px-4 py-2.5" %}
      {% for error in field.errors %}<p class="mt-1 text-sm text-accent">{{ error }}</p>{% endfor %}
    </div>
  {% endfor %}
  <button type="submit" class="rounded-lg bg-accent px-6 py-2.5 font-medium text-ink">Send</button>
</form>
```

Note: for Bootstrap form styling use `crispy-forms` + `crispy-bootstrap5` instead — see `bootstrap.md`.

## Messages framework

`templates/partials/_messages.html`:

```html
{% if messages %}
  <div class="mx-auto max-w-5xl px-6">
    {% for message in messages %}
      <div class="my-2 rounded-lg px-4 py-3 text-sm
        {% if message.tags == 'error' %}bg-red-950 text-red-200
        {% elif message.tags == 'success' %}bg-emerald-950 text-emerald-200
        {% else %}bg-stone-800 text-stone-200{% endif %}">
        {{ message }}
      </div>
    {% endfor %}
  </div>
{% endif %}
```
