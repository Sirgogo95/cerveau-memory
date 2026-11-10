# HTMX with Django

HTMX adds dynamic behavior to Django templates without a SPA. Views return small HTML partials that swap into the page.

## Include

In `base.html` (before `{% block extra_js %}`):

```html
<script src="https://unpkg.com/htmx.org@2"></script>
```

## CSRF wiring (required for POST/PUT/DELETE)

Django rejects unsafe requests without a CSRF token. Set it globally on the body so every HTMX request carries it:

```html
<body hx-headers='{"X-CSRFToken": "{{ csrf_token }}"}'>
```

Alternatively include `{% csrf_token %}` inside any `<form>` that HTMX submits — for non-form requests, the header approach above is cleanest.

## Returning partials from views

Keep a partial template and return just that fragment for HTMX requests:

```python
from django.shortcuts import render

def search(request):
    q = request.GET.get("q", "")
    results = Item.objects.filter(name__icontains=q)
    return render(request, "partials/_results.html", {"results": results})
```

Detect HTMX requests when a view serves both full pages and fragments:

```python
if request.headers.get("HX-Request"):
    return render(request, "partials/_results.html", ctx)
return render(request, "page.html", ctx)
```

## Common patterns

### Active search

```html
<input type="search" name="q"
  hx-get="{% url 'search' %}"
  hx-trigger="input changed delay:300ms, search"
  hx-target="#results"
  hx-indicator="#spinner"
  class="w-full rounded-lg border border-stone-700 bg-stone-900 px-4 py-2.5">
<div id="results"></div>
```

### Inline edit / replace

```html
<button hx-get="{% url 'edit' item.pk %}" hx-target="#item-{{ item.pk }}" hx-swap="outerHTML">
  Edit
</button>
```

The view returns the edit form partial; saving returns the read-only partial — both with the same `id` so the swap targets cleanly.

### Infinite scroll

```html
<tr hx-get="{% url 'page' next_page %}" hx-trigger="revealed" hx-swap="afterend">
  ...last row...
</tr>
```

### Delete

```html
<button hx-delete="{% url 'delete' item.pk %}"
        hx-target="#item-{{ item.pk }}"
        hx-swap="outerHTML swap:200ms"
        hx-confirm="Delete this item?">Delete</button>
```

### Form submit with validation errors

POST to a view that re-renders the form partial (with errors) on invalid, or returns the success partial on valid. Target the form's wrapping element with `hx-target` + `hx-swap="outerHTML"`.

## Swap strategies

- `innerHTML` (default), `outerHTML`, `beforeend`/`afterend`, `delete`, `none`.
- Add CSS transitions with `hx-swap="... swap:200ms settle:200ms"` and the `htmx-swapping`/`htmx-settling` classes for smooth animation.

## Tips

- Return the smallest fragment that does the job; keep partials in `templates/partials/`.
- Reuse the same partial inside the full-page template via `{% include %}` so first load and HTMX updates share markup.
- For richer client behavior, pair HTMX with Alpine.js (`x-data`, `x-show`) for purely client-side state like dropdowns and modals.
