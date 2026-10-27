#!/usr/bin/env python3
"""Resolve Investopedia term identifiers in the annotation into term names.

The glossary is not redistributed here (see README.md), so this takes your own
copy and writes a companion CSV with the names filled in.

usage: resolve_terms.py --glossary GLOSSARY.json --in IN.csv --out OUT.csv
"""
import argparse, csv, json, sys


def load_names(path):
    """Map term id -> term name, accepting either a list or an id-keyed dict."""
    data = json.load(open(path, encoding="utf-8"))
    items = data.values() if isinstance(data, dict) else data
    names = {}
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            continue
        tid = entry.get("id", entry.get("term_id", i))
        name = entry.get("term", entry.get("name", entry.get("title")))
        if name is not None:
            names[str(tid)] = name
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary", required=True)
    ap.add_argument("--in", dest="src", default="mmlu_finance_terms.csv")
    ap.add_argument("--out", dest="dst", default="mmlu_finance_terms_named.csv")
    a = ap.parse_args()

    names = load_names(a.glossary)
    if not names:
        sys.exit("no terms found in %s; expected a list or dict of term records" % a.glossary)

    missing, written = set(), 0
    with open(a.src, encoding="utf-8") as fh, open(a.dst, "w", newline="", encoding="utf-8") as out:
        r = csv.DictReader(fh)
        cols = list(r.fieldnames) + ["gold_inv_term_name"]
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        for row in r:
            resolved = []
            for t in [t for t in row["gold_inv_term_id"].split(";") if t.strip()]:
                if t in names:
                    resolved.append(names[t])
                else:
                    missing.add(t)
                    resolved.append("")
            row["gold_inv_term_name"] = ";".join(resolved)
            w.writerow(row)
            written += 1

    print("wrote %s (%d rows)" % (a.dst, written))
    if missing:
        print("%d identifiers had no match in the glossary, e.g. %s"
              % (len(missing), sorted(missing)[:5]))
        print("check that the glossary is the curated bank of Jhirad et al. (2023)")


if __name__ == "__main__":
    main()
