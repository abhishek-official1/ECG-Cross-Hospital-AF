# A3 portrait ECG research poster

Compile from this folder using:

```powershell
xelatex -interaction=nonstopmode -halt-on-error ecg_research_poster.tex
```

The PDF is designed at 297 x 420 mm (A3 portrait). Print at **Actual size / 100%**, without scaling. The editable source is `ecg_research_poster.tex`; vector chart files are in `figures/`. Run `python make_figures.py` to regenerate plots from the latest experiment artifacts. Arial is used for reproducible local typography.

Counts and plots use run `ecg_results_20261027_205834`. No author or institutional identity has been invented. The poster explicitly marks the exploratory Georgia evaluation, patient-identity limitations, unmet source sensitivity target and nonclinical status. Links in the reference list point to primary sources.
