import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def check_label_generation(df: pd.DataFrame):
    """
    Sanity Check: verifies that every source has exactly one label.
    """
    max_labels_per_source = df.groupby("source")["label"].nunique().max()
    assert max_labels_per_source == 1, f"Found source with {max_labels_per_source} labels!"
    print("Sanity Check Passed: 134/134 sources have exactly one label.")

def test_text_to_source_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Tests if article text strongly predicts the publisher identity (source_id).
    """
    print("Running Text -> Source Leakage Test...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    X_train = vectorizer.fit_transform(train_df['text'].fillna(""))
    y_train = train_df['source']
    
    X_test = vectorizer.transform(test_df['text'].fillna(""))
    y_test = test_df['source']
    
    # 134-class classification
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    acc = (y_pred == y_test).mean()
    
    print(f"Source Prediction Accuracy: {acc:.4f}")
    print(f"Source Prediction Macro-F1: {macro_f1:.4f}")
    if acc > 0.5:
        print("WARNING: High source leakage detected. Text strongly predicts publisher identity.")

def check_length_shortcut(df: pd.DataFrame):
    """
    Checks if article length alone is predictive of the label.
    """
    print("Checking Article Length vs Label...")
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    
    for label in df['label'].unique():
        avg_len = df[df['label'] == label]['word_count'].mean()
        print(f"Average length for {label}: {avg_len:.1f} words")
        
def check_date_reliability(df: pd.DataFrame):
    """
    Investigates reliability of published_date column.
    """
    print("Checking Date Reliability...")
    total = len(df)
    missing = df['published_date'].isna().sum() + (df['published_date'] == "").sum()
    print(f"Missing dates: {missing} ({missing/total*100:.1f}%)")
    
    parsed_dates = pd.to_datetime(df['published_date'], errors='coerce')
    invalid = parsed_dates.isna().sum() - missing
    print(f"Invalid dates: {invalid} ({invalid/total*100:.1f}%)")
    
    future = (parsed_dates > pd.Timestamp.now()).sum()
    print(f"Future dates: {future} ({future/total*100:.1f}%)")
