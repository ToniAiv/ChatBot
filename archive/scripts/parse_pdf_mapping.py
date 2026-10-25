"""
Parser για το ipiresies.pdf — χειροκίνητη αντιστοίχιση intents ↔ υπηρεσιών
==========================================================================
Το PDF κωδικοποιεί την πληροφορία με χρώματα. Υπόμνημα (από τον συντάκτη):

    ΚΟΚΚΙΝΗ υπηρεσία   → δεν έχει intent στο database   (πρέπει να ΠΡΟΣΤΕΘΕΙ)
    ΜΠΛΕ υπηρεσία      → έχει intent
    ΠΡΑΣΙΝΟ δίπλα σε μπλε → το όνομα του intent που αντιστοιχεί
    ΚΙΤΡΙΝΗ επισήμανση → το intent μοιάζει μόνο ΛΙΓΟ με την υπηρεσία
    ΠΡΑΣΙΝΟ στην αρχική λίστα → δεν υπάρχει ίδια υπηρεσία στη σελίδα

Το διάβασμα με το μάτι σε 8 σελίδες είναι επιρρεπές σε λάθη, οπότε τα χρώματα
εξάγονται από το ίδιο το PDF.

Χρώματα που βρέθηκαν:
    #ee0000 κόκκινο | #0563c1 & #4472c4 μπλε (link / visited link)
    #70ad47 πράσινο | #000000 μαύρο
Η κίτρινη επισήμανση δεν είναι annotation αλλά σχεδιασμένο ορθογώνιο,
οπότε ανιχνεύεται γεωμετρικά.

Χρήση:
    python3 parse_pdf_mapping.py --pdf ~/Documents/ipiresies.pdf
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import fitz

RED = 0xEE0000
BLUE = {0x0563C1, 0x4472C4}
GREEN = 0x70AD47
BLACK = 0x000000

# Η λίστα intents τελειώνει εδώ και αρχίζει ο κατάλογος υπηρεσιών
SERVICES_MARKER = "Δημότες"

_LIST_NUM = re.compile(r"^\s*(\d+)\.\s*")
_LIST_ALPHA = re.compile(r"^\s*([a-z])\.\s*")


def yellow_rects(page) -> list:
    out = []
    for d in page.get_drawings():
        f = d.get("fill")
        if f and len(f) == 3 and f[0] > 0.8 and f[1] > 0.8 and f[2] < 0.5:
            out.append(fitz.Rect(d["rect"]))
    return out


def _area(r) -> float:
    return max(0.0, r.x1 - r.x0) * max(0.0, r.y1 - r.y0)


def is_highlighted(span_rect, rects) -> bool:
    r = fitz.Rect(span_rect)
    for y in rects:
        inter = r & y
        if inter.is_valid and _area(inter) > 0.4 * _area(r):
            return True
    return False


def extract_lines(doc) -> list:
    """[(page_no, y, [(text, color, highlighted), ...]), ...] σε σειρά ανάγνωσης."""
    lines = []
    for pno, page in enumerate(doc, start=1):
        ys = yellow_rects(page)
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = []
                for s in line["spans"]:
                    if not s["text"].strip():
                        continue
                    spans.append((s["text"], s["color"],
                                  is_highlighted(s["bbox"], ys)))
                if spans:
                    lines.append((pno, line["bbox"][1], spans))
    return lines


def clean_intent(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(".,;")


def parse(doc) -> tuple:
    lines = extract_lines(doc)

    # ── Πού τελειώνει η λίστα intents ────────────────────────────────────────
    split_at = next(
        (i for i, (_, _, spans) in enumerate(lines)
         if any(SERVICES_MARKER in t for t, _, _ in spans)),
        len(lines),
    )

    # ── Μέρος Α: όλα τα intents του database ─────────────────────────────────
    # Η αρίθμηση δεν είναι πάντα στην ίδια γραμμή με το όνομα: στα στοιχεία
    # 100 και 101 το Word βγάζει το «100.» ως ξεχωριστό block. Οπότε δεν
    # απαιτούμε πρόθεμα αριθμού — απλώς πετάμε τις καθαρά αριθμητικές γραμμές.
    intents = []
    for _, _, spans in lines[:split_at]:
        text = "".join(t for t, _, _ in spans)
        name = clean_intent(_LIST_NUM.sub("", text))
        if not name or name.lower().startswith("intents from"):
            continue
        if re.fullmatch(r"[\d.\s]+", name):
            continue
        colors = {c for t, c, _ in spans
                  if t.strip() and not re.fullmatch(r"[\d.\s]+", t)}
        # Πράσινο = δεν υπάρχει ίδια υπηρεσία στη σελίδα
        status = "χωρίς_ίδια_υπηρεσία" if GREEN in colors else "έχει_υπηρεσία"
        intents.append({"intent": name, "status": status})

    # ── Μέρος Β: κατάλογος υπηρεσιών ─────────────────────────────────────────
    def is_annotation(color: int, highlighted: bool) -> bool:
        """
        Σχόλιο intent δίπλα σε υπηρεσία.

        Κανονικά είναι πράσινο, ΑΛΛΑ όταν ο συντάκτης το επισήμανε κίτρινο η
        γραμματοσειρά έμεινε μαύρη — οπότε όλα τα «χαλαρά» intents χάνονταν.
        Το μπλε/κόκκινο μένει τίτλος υπηρεσίας ακόμη κι αν το κίτρινο
        ορθογώνιο ξεχειλίζει πάνω του.
        """
        if color == GREEN:
            return True
        return highlighted and color not in BLUE and color != RED

    # Ένα span μπορεί να περιέχει πολλά intents χωρισμένα με κενά
    # («ταπ_ηλεκτροδότηση      μη_οφειλή_ταπ_οικόπεδα»). Τα ονόματα των
    # intents δεν περιέχουν ποτέ κενό, οπότε το split είναι ασφαλές.
    def annotations(spans_):
        toks = []
        pending = False   # το προηγούμενο span έκλεισε σε «_» → συνεχίζεται
        for t, c, hl in spans_:
            if not is_annotation(c, hl) and not pending:
                continue
            # Ο συντάκτης βάφει πράσινο μέχρι την τελευταία κάτω παύλα και
            # αφήνει την κατάληξη μαύρη («…στάθμευσης_» + «αμεα»). Χωρίς αυτό
            # το intent κόβεται στη μέση.
            pending = t.rstrip().endswith("_")
            for tok in t.split():
                tok = clean_intent(tok)
                # Πέτα αριθμούς και γράμματα λίστας («a.», «g.») που
                # τυχαίνει να πέφτουν μέσα στην κίτρινη επισήμανση
                if not tok or re.fullmatch(r"[\d.\s]+|[a-zA-Z]", tok):
                    continue
                toks.append((tok, hl))

        # Μεγάλα ονόματα σπάνε στο τέλος της γραμμής
        # («…_θέσης_στάθμευσης_» + «αμεα») — ένωσέ τα ξανά.
        merged = []
        for tok, hl in toks:
            if merged and (merged[-1][0].endswith("_") or tok.startswith("_")):
                merged[-1] = (merged[-1][0] + tok, merged[-1][1] or hl)
            else:
                merged.append((tok, hl))
        return [(t, hl) for t, hl in merged if t.strip("_")]

    services, section = [], ""
    for _, _, spans in lines[split_at:]:
        text = "".join(t for t, _, _ in spans)
        stripped = text.strip()
        if not stripped:
            continue

        non_green = [(t, c) for t, c, hl in spans
                     if not is_annotation(c, hl) and t.strip()
                     and not _LIST_ALPHA.match(t.strip())]
        colors = {c for _, c in non_green}

        # Επικεφαλίδα ενότητας: αριθμημένη γραμμή χωρίς μπλε/κόκκινο κείμενο.
        # Μπορεί να κουβαλά και σχόλια intent («34. Τεχνική υπηρεσία
        # τεχνική_υπηρεσία …») — αυτά αφορούν ΟΛΗ την ενότητα, όχι μία σελίδα.
        if _LIST_NUM.match(stripped) and not (colors & BLUE) and RED not in colors:
            section = clean_intent(_LIST_NUM.sub("", "".join(
                t for t, c, hl in spans if not is_annotation(c, hl))))
            head = annotations(spans)
            if head:
                services.append({
                    "section": section,
                    "service": f"[ΕΝΟΤΗΤΑ] {section}",
                    "has_intent": True,
                    "intents": head,
                })
            continue

        has_service = bool(colors & BLUE) or RED in colors
        starts_item = bool(_LIST_ALPHA.match(stripped))

        # Χρωματιστή γραμμή ΧΩΡΙΣ γράμμα λίστας = συνέχεια τίτλου που έσπασε
        # («…για τοποθέτηση περίφραξης» / «σε οδόστρωμα»). Χωρίς αυτό, κάθε
        # δεύτερη γραμμή μετριόταν ως ξεχωριστή υπηρεσία.
        if not starts_item:
            if services:
                if has_service:
                    tail = clean_intent("".join(
                        t for t, c, hl in spans if not is_annotation(c, hl)))
                    if tail:
                        services[-1]["service"] = clean_intent(
                            services[-1]["service"] + " " + tail)
                services[-1]["intents"].extend(annotations(spans))
            continue

        title = clean_intent(_LIST_ALPHA.sub("", "".join(
            t for t, c, hl in spans if not is_annotation(c, hl))))
        attached = annotations(spans)

        services.append({
            "section": section,
            "service": title,
            "has_intent": RED not in colors,
            "intents": attached,
        })

    return intents, services


def main() -> None:
    ap = argparse.ArgumentParser(description="Parser του ipiresies.pdf")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--intents-out", default="pdf_intents.csv")
    ap.add_argument("--services-out", default="pdf_services.csv")
    args = ap.parse_args()

    path = Path(args.pdf).expanduser()
    if not path.exists():
        sys.exit(f"Δεν βρέθηκε: {path}")

    doc = fitz.open(str(path))
    intents, services = parse(doc)

    with open(args.intents_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["intent", "status"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(intents)

    with open(args.services_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["ενότητα", "υπηρεσία", "έχει_intent", "intents", "χαλαρή"],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        for s in services:
            w.writerow({
                "ενότητα": s["section"],
                "υπηρεσία": s["service"],
                "έχει_intent": "ναι" if s["has_intent"] else "ΟΧΙ",
                "intents": ", ".join(i for i, _ in s["intents"]),
                "χαλαρή": "ναι" if any(hl for _, hl in s["intents"]) else "",
            })

    n_unmatched = sum(1 for i in intents if i["status"] == "χωρίς_ίδια_υπηρεσία")
    n_red = sum(1 for s in services if not s["has_intent"])
    n_loose = sum(1 for s in services if any(hl for _, hl in s["intents"]))
    mapped = {i for s in services for i, _ in s["intents"]}

    print("═" * 66)
    print("  ΑΝΑΓΝΩΣΗ ΤΟΥ PDF")
    print("═" * 66)
    print(f"  Intents στη λίστα            : {len(intents)}")
    print(f"    ├ με αντίστοιχη υπηρεσία   : {len(intents) - n_unmatched}")
    print(f"    └ ΧΩΡΙΣ ίδια υπηρεσία      : {n_unmatched}")
    print(f"\n  Υπηρεσίες στον κατάλογο      : {len(services)}")
    print(f"    ├ με intent (μπλε)         : {len(services) - n_red}")
    print(f"    └ ΧΩΡΙΣ intent (κόκκινες)  : {n_red}   ← προς ΠΡΟΣΘΗΚΗ")
    print(f"\n  Υπηρεσίες με «χαλαρό» intent : {n_loose}   ← προς ΕΛΕΓΧΟ")
    print(f"  Μοναδικά intents σε σχόλια   : {len(mapped)}")
    print(f"\n  💾 {args.intents_out}\n  💾 {args.services_out}")
    print("═" * 66)


if __name__ == "__main__":
    main()
