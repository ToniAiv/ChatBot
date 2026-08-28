import urllib.request, re
from urllib.parse import urljoin

headers = {"User-Agent": "MunicipalityCrawler/2.0"}

def fetch(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode("utf-8", errors="replace"), r.geturl()

html, _ = fetch("https://eservices.heraklion.gr/")

# Βγάλε όλα τα hrefs
all_hrefs = re.findall(r'href=["\']([^"\'#\s]+)["\']', html)

# Κανονικοποίησε σε absolute
base = "https://eservices.heraklion.gr/"
absolute = list(dict.fromkeys(
    urljoin(base, h) for h in all_hrefs
    if not h.startswith(("mailto:", "tel:", "javascript:"))
))

# Χώρισε ανά domain
eserv  = [u for u in absolute if "eservices.heraklion.gr" in u]
herakl = [u for u in absolute if "heraklion.gr" in u and "eservices" not in u]
other  = [u for u in absolute if "heraklion.gr" not in u]

print(f"=== eservices.heraklion.gr links ({len(eserv)}) ===")
for u in sorted(eserv):
    print(f"  {u}")

print(f"\n=== www.heraklion.gr links ({len(herakl)}) ===")
for u in sorted(herakl)[:20]:
    print(f"  {u}")

print(f"\n=== External links ({len(other)}) ===")
for u in sorted(other)[:5]:
    print(f"  {u}")
