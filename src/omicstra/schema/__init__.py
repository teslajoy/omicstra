"""omicstra biomedical schema.

the YAML file is the schema source. python dataclasses and JSON Schema are
generated from it at build time (not at import time). linkml is a build-time
dependency only: it lives in [dev] extras and is never imported here.

to regenerate:
    gen-pydantic src/omicstra/schema/omicstra.yaml > src/omicstra/schema/_generated.py
    gen-jsonschema src/omicstra/schema/omicstra.yaml > src/omicstra/schema/omicstra.schema.json
"""
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "omicstra.yaml"