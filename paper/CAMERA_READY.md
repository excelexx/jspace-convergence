# Camera-ready version

`neurips_2026_camera_ready.tex` and `neurips_2026_camera_ready.pdf` are the
camera-ready duplicates. The original submission files are unchanged.

The duplicate enables the NeurIPS workshop `final` option, restores the authors
and affiliations preserved in the submission source, uses the public GitHub code
link, and hides hyperlink borders. Its separate checklist updates the code-release
wording. Scientific content, results, figures, and references are unchanged.

Build from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error neurips_2026_camera_ready.tex
bibtex neurips_2026_camera_ready
pdflatex -interaction=nonstopmode -halt-on-error neurips_2026_camera_ready.tex
pdflatex -interaction=nonstopmode -halt-on-error neurips_2026_camera_ready.tex
```
