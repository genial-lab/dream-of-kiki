# arXiv submission package — Paper 1

**Generated** : 2026-04-18 via pandoc 3.9.0.2
**Source** : `docs/papers/paper1/full-draft.md`
**Output** : `build/full-draft.tex` (408 lines, single-file
LaTeX article class)

## Files in this build

| File | Purpose |
|------|---------|
| `full-draft.tex` | Main LaTeX source for arXiv upload |
| `README-arxiv.md` | This file (submission instructions) |
| `.gitignore` | Ignore LaTeX build artifacts |

## arXiv submission steps

1. **Verify LaTeX compiles** locally (optional but recommended) :
   ```bash
   # Requires LaTeX installed (e.g., brew install basictex)
   cd docs/papers/paper1/build
   pdflatex full-draft.tex
   pdflatex full-draft.tex   # second pass for refs
   ```
   If you don't have LaTeX, **skip this step** — arXiv compiles
   the PDF on their server.

2. **Login to arXiv** : https://arxiv.org/submit
3. **Choose categories** :
   - Primary : `cs.LG` (Machine Learning)
   - Cross-list : `q-bio.NC` (Neurons and Cognition), `cs.AI`
4. **Upload** :
   - `full-draft.tex` (main source)
   - `../references.bib` (bibliography — copy into build/ if
     arXiv requires single directory)
   - Figures (none yet — to be added before final submission)
5. **arXiv preview** : verify rendered PDF before final submit
6. **Submit** → receive arXiv ID `2604.XXXXX`

## Re-render command

If `full-draft.md` is updated, re-render :
```bash
cd docs/papers/paper1
pandoc full-draft.md -o build/full-draft.tex \
    --bibliography=references.bib --citeproc --standalone
```

## Known limitations

- `full-draft.md` includes `[INCLUDE: ...]` directives that
  pandoc does NOT auto-resolve. Current build inlines those
  literals in the .tex output. To inline section files
  properly, either :
  - Manually concatenate section .md files into `full-draft.md`
    before render (preserves source-of-truth pattern), OR
  - Use `pandoc-include` filter (`pip install pandoc-include`)
    to auto-resolve `[INCLUDE:]` directives.
- Figures are absent. Add via `\includegraphics{figures/X.pdf}`
  in section files + ship in build/ for arXiv.
- BibTeX rendering uses `--citeproc` (built-in) rather than
  proper `bibtex` + .bbl ; for production submission consider
  `pandoc --natbib` or `--biblatex` with manual bibtex pass.

## Pre-submission checklist

- [ ] All synthetic-data caveats explicit in §7 + §8
- [ ] OSF DOI inserted in §6.1 (currently pending OSF lock)
- [ ] References.bib resolves all `\cite{}` calls
- [ ] Authorship byline `Saillant, Clément`
- [ ] Repo URL present in §5.5
- [ ] HuggingFace model URLs present
- [ ] Zenodo DOI inserted (post-mint)
- [ ] No AI attribution
- [ ] Figures embedded
- [ ] LaTeX compiles cleanly (if local LaTeX available)
