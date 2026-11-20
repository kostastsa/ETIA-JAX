"""Write plate.html from plate_template.html and the STE texts of howto.md."""
import json
import re
from pathlib import Path

D = Path(__file__).parent
texts = {}
for m in re.finditer(r"## (.+?)\n\n(.*?)(?=\n## |\Z)", (D / "howto.md").read_text(), re.S):
    texts[m.group(1).strip()] = "".join(f"<p>{p.strip()}</p>" for p in m.group(2).strip().split("\n\n"))
html = (D / "plate_template.html").read_text().replace("@HOWTO@", json.dumps(texts, ensure_ascii=False, indent=1))
(D / "plate.html").write_text(html)
print("wrote plate.html with", len(texts), "texts")
