FACE_DETECTION_DATA = ""

all: clean data/face-detection.csv

clean:
	rm -rf data

data/face-detection.csv:
	python scripts/download.py $(FACE_DETECTION_DATA) $@

dependencies:
	pip install -r requirements.txt

.PHONY: all clean dependencies