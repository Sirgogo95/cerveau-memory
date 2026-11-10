# Bootstrap in Django (fallback)

Use Bootstrap only when the user asks for it, the project already uses it, or a ready-made component library via CDN is explicitly preferred over Tailwind utilities.

## CDN (no build)

In `base.html`:

```html
<head>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
  ...
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
```

## django-bootstrap5

```bash
pip install django-bootstrap5
```

`settings.py` → add `"django_bootstrap5"` to `INSTALLED_APPS`. Then in templates:

```html
{% load django_bootstrap5 %}
{% bootstrap_css %}
{% bootstrap_javascript %}

<form method="post">
  {% csrf_token %}
  {% bootstrap_form form %}
  {% bootstrap_button button_type="submit" content="Save" %}
</form>
```

`{% bootstrap_form %}` renders all fields with Bootstrap classes, labels, help text, and errors automatically.

## Crispy forms (most control over Bootstrap form layout)

```bash
pip install crispy-forms crispy-bootstrap5
```

`settings.py`:

```python
INSTALLED_APPS += ["crispy_forms", "crispy_bootstrap5"]
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"
```

Template:

```html
{% load crispy_forms_tags %}
<form method="post">{% csrf_token %}{{ form|crispy }}<button class="btn btn-primary">Save</button></form>
```

For programmatic layouts (rows, columns, fieldsets), use a `FormHelper` with `Layout`, `Row`, `Column` in the form class.

## Design note

Bootstrap's default look is recognizable and generic. If the user cares about a distinctive aesthetic, still apply the design principles in `design-aesthetics.md`: override Bootstrap's CSS variables (`--bs-primary`, `--bs-body-font-family`, etc.), bring in distinctive fonts, and customize beyond the default theme rather than shipping stock Bootstrap.
