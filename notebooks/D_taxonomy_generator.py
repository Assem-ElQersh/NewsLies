import pandas as pd
import re

df = pd.read_csv('experiments/taxonomy/arabert_errors_sample.csv')

def categorize_error(row):
    title = str(row.get('title', ''))
    text = str(row.get('text', ''))
    full_text = title + " " + text
    
    # 1. Source-Specific Artifacts (e.g. newspaper names, specific markers)
    source_artifacts = ['النهار أونلاين', 'وكالة الأنباء', 'رويترز', 'CNN', 'الجزيرة', 'العربية', 'عاجل']
    if any(artifact in full_text for artifact in source_artifacts):
        return "Source-Specific Artifacts / Boilerplate"
        
    # 2. Government & Official Denials
    gov_keywords = ['نفى', 'تنفي', 'الحكومة', 'رئيس', 'وزير', 'الرئيس', 'مجلس الوزراء', 'رسمي', 'وزارة']
    if any(word in full_text for word in gov_keywords):
        return "Government & Official Statements"
        
    # 3. Health & COVID-19
    health_keywords = ['كورونا', 'تلقيح', 'لقاح', 'صحة', 'مستشفى', 'فيروس', 'وفاة', 'إصابة']
    if any(word in full_text for word in health_keywords):
        return "Health & COVID-19"
        
    # 4. Economics & Pricing
    econ_keywords = ['دينار', 'دولار', 'أسعار', 'اقتصاد', 'مالية', 'بنك', 'سوق']
    if any(word in full_text for word in econ_keywords):
        return "Economics & Pricing"
        
    # 5. Sports
    sports_keywords = ['ملعب', 'مباراة', 'فريق', 'بطولة', 'كأس', 'نادي', 'لاعب']
    if any(word in full_text for word in sports_keywords):
        return "Sports News"
        
    # 6. Crime & Accidents
    crime_keywords = ['شرطة', 'أمن', 'اعتقال', 'جريمة', 'حادث', 'محكمة']
    if any(word in full_text for word in crime_keywords):
        return "Crime & Accidents"

    return "General/Other Ambiguous Context"

df['error_category'] = df.apply(categorize_error, axis=1)

# Save categorized
df.to_csv('experiments/taxonomy/categorized_errors.csv', index=False)

# Generate Summary Report
summary = df.groupby(['error_category', 'label', 'pred']).size().reset_index(name='count')
summary = summary.sort_values(by=['error_category', 'count'], ascending=[True, False])

with open('experiments/taxonomy/taxonomy_report.md', 'w', encoding='utf-8') as f:
    f.write("# Error Taxonomy Summary (500 Samples)\n\n")
    f.write("This taxonomy analyzes 500 errors from the source-disjoint split. The AraBERT model achieved 37% Macro-F1 on this split, indicating a complete failure to generalize when publisher clues are removed.\n\n")
    f.write("## Category Breakdown\n")
    
    cat_counts = df['error_category'].value_counts()
    for cat, count in cat_counts.items():
        f.write(f"- **{cat}**: {count} errors\n")
        
    f.write("\n## Detailed Misclassifications by Category\n")
    f.write("| Error Category | True Label | Predicted Label | Count |\n")
    f.write("| :--- | :--- | :--- | :--- |\n")
    for _, row in summary.iterrows():
        f.write(f"| {row['error_category']} | {row['label']} | {row['pred']} | {row['count']} |\n")

print("Taxonomy generated and saved.")
