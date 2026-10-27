#!/usr/bin/env python3
"""Reproduce the coverage numbers quoted in the paper and in README.md.

usage: summarise.py [mmlu_finance_terms.csv]
"""
import collections, csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else "mmlu_finance_terms.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))

ids = [[t for t in r["gold_inv_term_id"].split(";") if t.strip()] for r in rows]
bearing = [t for t in ids if t]
pairs = sum(len(t) for t in bearing)

print("questions                  %5d" % len(rows))
print("  term-bearing             %5d" % len(bearing))
print("  zero-term controls       %5d" % (len(rows) - len(bearing)))
print("(question, term) pairs     %5d" % pairs)
print("mean terms per term-bearing %.2f" % (pairs / len(bearing)))

per = collections.Counter(len(t) for t in bearing)
print("\nterms per question")
for k in sorted(per):
    print("  %d  %4d  %5.1f%%" % (k, per[k], 100 * per[k] / len(bearing)))

per_sub = collections.Counter(r["subject"] for r in rows)
print("\nquestions per subject")
for s, n in sorted(per_sub.items(), key=lambda kv: -kv[1]):
    print("  %-30s %4d" % (s, n))
