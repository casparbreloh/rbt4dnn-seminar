# Slide design system

The presentation is intentionally one zero-build HTML document at `slides/index.html`. GitHub
Pages publishes it at one canonical URL. A fragment such as `#8` identifies a slide for sharing,
but there are no routes, generated pages, or framework dependencies.

## Visual thesis

The system combines three observed keynote traits:

- OpenAI-style directness: oversized sans-serif statements, sparse signal color, and one dominant
  idea per slide.
- Linear-style craft: strict alignment, thin rules, restrained motion, and diagrams drawn on the
  canvas instead of placed inside dashboard cards.
- Anthropic-style warmth: an off-white paper ground, near-black ink, and occasional editorial
  pacing rather than a uniformly technical interface.

These are directional references, not copied templates or brand assets. The deck uses no company
logos or proprietary slide assets.

## Tokens

All global decisions are CSS variables in `:root`:

- `--cream`, `--paper`, and `--ink` define the neutral canvas.
- `--blue` marks evidence and positive comparison.
- `--orange` marks warnings, gaps, and the core critique.
- `--font-sans` is the display/body family; `--font-mono` is only for metadata and labels.
- `--pad-x` and `--pad-y` establish the shared 1920 × 1080 page margin.

## Primitives

Use these before creating a new class:

| Primitive | Purpose |
| --- | --- |
| `.frame` | Shared slide margin and vertical structure |
| `.topline` | Section label, slide context, and top hairline |
| `.display`, `.headline`, `.statement` | Three deliberate headline scales |
| `.cols-2` | Open two-column composition without a container |
| `.rule-grid` + `.rule-cell` | Comparable items separated only by hairlines |
| `.rail` + `.rail-row` | Ordered arguments, definitions, or recommendations |
| `.stat` | One number with one explanatory label |
| `.sequence` | A flat four-step process |
| `.bars` | Direct data bars without axes or chart furniture |
| `.axis` | Sparse conceptual plot |
| `.diagram` | Shared SVG grammar for flows, targets, gates, and cycles |
| `.compare-board` | Metric comparison paired with target-distribution diagrams |
| `.scenario-map`, `.scope-strip` | Flat reading-density strips for contrasts and research questions |
| `.large-bars`, `.chart-kicker` | Larger evidence charts with an explicit reading frame |
| `.figure` | A single experiment image with a provenance label |
| `.footer` | Source/context line and page number |

Every slide and the surrounding letterbox use the same cream paper canvas. Blue and orange are
signals inside diagrams and metrics, never alternate slide themes. Rounded rectangles, shadows,
badges, nested panels, and decorative gradients are excluded from the slide system. The small
navigation and editing controls sit outside the authored stage and are not presentation content.

## Editing rules

1. Keep one dominant idea per slide.
2. Prefer a rule, whitespace, or direct diagram over a bordered container.
3. Use at most one primary diagram or figure on a slide.
4. Use roughly two-thirds of the content field unless the slide is an intentional pacing beat.
5. Add density through labels, comparisons, annotations, and evidence—not ornamental panels.
6. Add a new primitive only when at least three slides can reuse it.
7. Keep content inside the fixed 1920 × 1080 stage; phone views scale rather than reflow.
8. Press `E` in the browser for local text editing. Those edits remain local until exported.
