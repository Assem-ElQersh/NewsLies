# Error Taxonomy Summary (500 Samples)

This taxonomy analyzes 500 errors from the source-disjoint split. The AraBERT model achieved 37% Macro-F1 on this split, indicating a complete failure to generalize when publisher clues are removed.

## Category Breakdown
- **Government & Official Statements**: 238 errors
- **Source-Specific Artifacts / Boilerplate**: 108 errors
- **General/Other Ambiguous Context**: 52 errors
- **Health & COVID-19**: 44 errors
- **Crime & Accidents**: 20 errors
- **Economics & Pricing**: 19 errors
- **Sports News**: 19 errors

## Detailed Misclassifications by Category
| Error Category | True Label | Predicted Label | Count |
| :--- | :--- | :--- | :--- |
| Crime & Accidents | credible | undecided | 12 |
| Crime & Accidents | credible | not credible | 3 |
| Crime & Accidents | not credible | credible | 3 |
| Crime & Accidents | not credible | undecided | 2 |
| Economics & Pricing | credible | not credible | 7 |
| Economics & Pricing | credible | undecided | 5 |
| Economics & Pricing | not credible | credible | 3 |
| Economics & Pricing | not credible | undecided | 3 |
| Economics & Pricing | undecided | not credible | 1 |
| General/Other Ambiguous Context | credible | undecided | 24 |
| General/Other Ambiguous Context | not credible | undecided | 11 |
| General/Other Ambiguous Context | credible | not credible | 8 |
| General/Other Ambiguous Context | not credible | credible | 7 |
| General/Other Ambiguous Context | undecided | not credible | 2 |
| Government & Official Statements | credible | undecided | 111 |
| Government & Official Statements | credible | not credible | 71 |
| Government & Official Statements | not credible | credible | 21 |
| Government & Official Statements | undecided | not credible | 14 |
| Government & Official Statements | not credible | undecided | 13 |
| Government & Official Statements | undecided | credible | 8 |
| Health & COVID-19 | credible | not credible | 17 |
| Health & COVID-19 | credible | undecided | 15 |
| Health & COVID-19 | not credible | undecided | 6 |
| Health & COVID-19 | not credible | credible | 4 |
| Health & COVID-19 | undecided | credible | 1 |
| Health & COVID-19 | undecided | not credible | 1 |
| Source-Specific Artifacts / Boilerplate | undecided | not credible | 37 |
| Source-Specific Artifacts / Boilerplate | undecided | credible | 25 |
| Source-Specific Artifacts / Boilerplate | credible | undecided | 24 |
| Source-Specific Artifacts / Boilerplate | credible | not credible | 16 |
| Source-Specific Artifacts / Boilerplate | not credible | credible | 3 |
| Source-Specific Artifacts / Boilerplate | not credible | undecided | 3 |
| Sports News | credible | undecided | 7 |
| Sports News | credible | not credible | 3 |
| Sports News | not credible | credible | 3 |
| Sports News | not credible | undecided | 3 |
| Sports News | undecided | not credible | 2 |
| Sports News | undecided | credible | 1 |
