#!/usr/bin/env bash
# Build report/main.pdf.
# main.tex follows the course template (pdfLaTeX: T2A + inputenc) and compiles as is on Overleaf / TeX Live:
#   pdflatex main && bibtex main && pdflatex main && pdflatex main
# Without TeX Live, this script uses tectonic (XeTeX) on a copy with a Unicode font instead of T2A/inputenc.
set -euo pipefail
cd "$(dirname "$0")"
if command -v pdflatex >/dev/null; then
  pdflatex -interaction=nonstopmode main >/dev/null; bibtex main >/dev/null
  pdflatex -interaction=nonstopmode main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null
  exit 0
fi
TECTONIC=${TECTONIC:-tectonic}
tmp=$(mktemp -d)
cp -r figures tables lit.bib "$tmp/"
sed -e 's/^\\usepackage{mathtext}//' \
    -e 's/^\\usepackage\[T2A\]{fontenc}/\\usepackage{fontspec}\\setmainfont{cmunrm}[Extension=.otf,BoldFont=cmunbx,ItalicFont=cmunti,BoldItalicFont=cmunbi]\\setmonofont{cmuntt}[Extension=.otf]/' \
    -e 's/^\\usepackage\[utf8\]{inputenc}//' main.tex > "$tmp/main.tex"
(cd "$tmp" && "$TECTONIC" -X compile main.tex)
cp "$tmp/main.pdf" main.pdf
