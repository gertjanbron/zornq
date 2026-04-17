# ZornQ paper — Makefile
# --------------------------------------------------------------------
# Usage:
#   make           # default: full build (pdflatex + biber + pdflatex x2)
#   make quick     # single pdflatex pass (no bibliography refresh)
#   make figures   # (re)generate figures via the Python pipeline
#   make clean     # remove auxiliary files
#   make distclean # remove auxiliary files + PDF
#   make watch     # continuous rebuild on file change (requires latexmk)
# --------------------------------------------------------------------

# ---- Config ----
TEXNAME      ?= main
TEX          ?= pdflatex
BIB          ?= biber
LATEXMK      ?= latexmk
PYTHON       ?= python3
CODE_DIR     ?= ../../code
FIG_SCRIPT   ?= $(CODE_DIR)/b177_figure_pipeline.py
DATA_DIR     ?= data
FIG_DIR      ?= figures

TEX_FLAGS    ?= -interaction=nonstopmode -halt-on-error -file-line-error

# ---- Phony ----
.PHONY: all quick figures data clean distclean watch help

# ---- Default target ----
all: figures $(TEXNAME).pdf

# ---- Full build: pdflatex + biber + pdflatex x2 ----
$(TEXNAME).pdf: $(TEXNAME).tex refs.bib | figures
	$(TEX) $(TEX_FLAGS) $(TEXNAME).tex
	$(BIB) $(TEXNAME)
	$(TEX) $(TEX_FLAGS) $(TEXNAME).tex
	$(TEX) $(TEX_FLAGS) $(TEXNAME).tex

# ---- Single-pass build (development) ----
quick:
	$(TEX) $(TEX_FLAGS) $(TEXNAME).tex

# ---- Figure generation ----
figures:
	@mkdir -p $(FIG_DIR) $(DATA_DIR)
	$(PYTHON) $(FIG_SCRIPT) --out-fig $(FIG_DIR) --out-data $(DATA_DIR)

# ---- Generate JSON benchmark data only ----
data:
	@mkdir -p $(DATA_DIR)
	$(PYTHON) $(FIG_SCRIPT) --out-data $(DATA_DIR) --no-figures

# ---- Continuous rebuild (requires latexmk) ----
watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(TEXNAME).tex

# ---- Cleanup ----
clean:
	rm -f *.aux *.log *.bbl *.bcf *.blg *.out *.toc *.lof *.lot \
	      *.run.xml *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm \
	      *.vrb *.xdv

distclean: clean
	rm -f $(TEXNAME).pdf
	rm -f $(FIG_DIR)/*.pdf $(FIG_DIR)/*.png $(FIG_DIR)/*.tex

help:
	@echo "ZornQ paper build targets:"
	@echo "  make           - build PDF with bibliography (default)"
	@echo "  make quick     - single pdflatex pass (no bib)"
	@echo "  make figures   - (re)generate figures via Python pipeline"
	@echo "  make data      - regenerate benchmark JSON only"
	@echo "  make watch     - continuous rebuild (latexmk -pvc)"
	@echo "  make clean     - remove aux files"
	@echo "  make distclean - remove aux + PDF + figures"
