# Web Collection Module

This module supports research-grade web-image candidate curation for TyreVisionX.

Implemented now:
- editable query catalog support
- manual CSV/JSON URL provider
- provider interface stubs for future approved search APIs
- source-specific adapter scaffolds for Wikimedia Commons, Pexels, Unsplash, Flickr, and manual Google discovery
- license/source metadata fields for candidate provenance
- candidate metadata schema
- local/URL download metadata update script
- image validation, exact hashing, perceptual hashing, blur checks, and quality filtering
- human-review queue generation

Not implemented now:
- Google HTML scraping
- automatic labeling
- bulk provider downloads
- production rights management
- validated anomaly scoring on web candidates

Use web candidates as research inputs that require filtering and human review.
