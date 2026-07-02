
# Makefile για crawler e-Υπηρεσιών Ηρακλείου v2.0

PY       ?= python3
CRAWLER  ?= crawler_script.py
DATA_DIR ?= data
OUT      ?= $(DATA_DIR)/heraklion_eservices.json
DEPTH    ?= 2

.PHONY: heraklion_eservices fresh clean install

# Εγκατάσταση εξαρτήσεων
install:
	pip install aiohttp beautifulsoup4 requests

# Crawl με resume support (συνεχίζει από checkpoint αν υπάρχει)
heraklion_eservices:
	@mkdir -p $(DATA_DIR)
	$(PY) $(CRAWLER) \
		--profile heraklion_eservices \
		--out $(OUT) \
		--mode sitemap-expand \
		--expand-depth $(DEPTH) \
		--resume

# Καθαρό crawl — διαγράφει checkpoint και ξεκινά από την αρχή
fresh:
	@mkdir -p $(DATA_DIR)
	rm -f $(DATA_DIR)/*_checkpoint.json $(DATA_DIR)/*_htmlstore.json.gz
	$(PY) $(CRAWLER) \
		--profile heraklion_eservices \
		--out $(OUT) \
		--mode sitemap-expand \
		--expand-depth $(DEPTH)

# Καθάρισμα αποτελεσμάτων και checkpoints
clean:
	rm -f $(DATA_DIR)/*.json $(DATA_DIR)/*.gz
	@echo "Καθαρίστηκαν όλα τα αρχεία στο $(DATA_DIR)/"
