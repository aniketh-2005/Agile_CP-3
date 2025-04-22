import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sqlite3
import hashlib 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.cluster import KMeans

# Initialize SQLite database
# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("model_results.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
                        hash TEXT,
                        model TEXT,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        conf_matrix TEXT,
                        per_class_precision TEXT,
                        per_class_recall TEXT,
                        per_class_f1 TEXT
                    )''')
    conn.commit()
    
    # Check if table was created and print columns
    cursor.execute("PRAGMA table_info(metrics);")
    columns = cursor.fetchall()
    if columns:
        print("✅ Table 'metrics' ensured in database.")
        for column in columns:
            print(f"Column: {column[1]}")
    
    conn.close()



# Hash the dataframe to detect duplicates
def hash_dataframe(df):
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return df_hash

# Save metrics to the database
def save_metrics_to_db(df, model_name, metrics_dict, conf_matrix, per_class_precision, per_class_recall, per_class_f1):
    df_hash = hash_dataframe(df)
    conn = sqlite3.connect("model_results.db")
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO metrics 
                      (hash, model, accuracy, precision, recall, f1_score, conf_matrix, per_class_precision, per_class_recall, per_class_f1)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                          df_hash, model_name,
                          metrics_dict['accuracy'],
                          metrics_dict['precision'],
                          metrics_dict['recall'],
                          metrics_dict['f1_score'],
                          ','.join(map(str, conf_matrix.flatten())),
                          ','.join(map(str, per_class_precision)),
                          ','.join(map(str, per_class_recall)),
                          ','.join(map(str, per_class_f1))
                      ))
    conn.commit()
    conn.close()

# Load metrics from the database if previously computed
def load_metrics_if_exist(df, model_name):
    df_hash = hash_dataframe(df)
    conn = sqlite3.connect("model_results.db")
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM metrics WHERE hash=? AND model=?', (df_hash, model_name))
    row = cursor.fetchone()
    conn.close()
    if row:
        _, _, accuracy, precision, recall, f1_score, conf_matrix_str, p_precision_str, p_recall_str, p_f1_str = row
        size = int(np.sqrt(len(conf_matrix_str.split(','))))
        conf_matrix = np.array(list(map(int, conf_matrix_str.split(',')))).reshape(size, size)
        per_class_precision = np.array(list(map(float, p_precision_str.split(','))))
        per_class_recall = np.array(list(map(float, p_recall_str.split(','))))
        per_class_f1 = np.array(list(map(float, p_f1_str.split(','))))
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "conf_matrix": conf_matrix,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1
        }
    return None

# Main app
def main():
    st.title("🧠 Multi-Model Classification with Deep Learning")
    init_db()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 **Data Preview:**")
        st.dataframe(df.head())

        all_columns = df.columns.tolist()
        target_column = st.selectbox("🎯 Select Target Column", all_columns)

        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            def is_text_column(series):
                if series.dtype == object:
                    sample = series.dropna().head(10)
                    return any(isinstance(x, str) and any(c.isalpha() for c in str(x)) for x in sample)
                return False

            def is_categorical_text(series):
                unique_count = series.dropna().nunique()
                return 1 < unique_count < 20

            def is_categorical_numeric(series):
                return pd.api.types.is_numeric_dtype(series) and 1 < series.nunique() < 20

            text_cols = [col for col in X.columns if is_text_column(X[col])]
            cat_text_cols = [col for col in text_cols if is_categorical_text(X[col])]
            tfidf_text_cols = [col for col in text_cols if col not in cat_text_cols]
            cat_num_cols = [col for col in X.columns if is_categorical_numeric(X[col]) and col not in text_cols]

            final_features = []
            feature_info = []

            for col in cat_text_cols:
                le = LabelEncoder()
                col_data = X[col].fillna('')
                encoded = le.fit_transform(col_data)
                final_features.append(pd.DataFrame(encoded, columns=[f"cattext_{col}"]))
                feature_info.append(f"Text (categorical - label encoded): {col}")

            if cat_num_cols:
                num_cat_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', MinMaxScaler())
                ])
                num_cat_data = num_cat_transformer.fit_transform(X[cat_num_cols])
                num_cat_df = pd.DataFrame(num_cat_data, columns=[f"catnum_{col}" for col in cat_num_cols])
                final_features.append(num_cat_df)
                feature_info.append(f"Numeric categorical (min-max scaled): {', '.join(cat_num_cols)}")

            for col in tfidf_text_cols:
                tfidf = TfidfVectorizer(max_features=100)
                col_data = X[col].fillna('')
                tfidf_result = tfidf.fit_transform(col_data)
                tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=[f"tfidf_{col}_{word}" for word in tfidf.get_feature_names_out()])
                final_features.append(tfidf_df)
                feature_info.append(f"Text (TF-IDF vectorized): {col} ({tfidf_df.shape[1]} features)")

            processed_cat_cols = cat_text_cols + cat_num_cols
            pure_num_cols = [col for col in X.select_dtypes(include=np.number).columns if col not in processed_cat_cols]

            if pure_num_cols:
                numeric_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                numeric_data = numeric_transformer.fit_transform(X[pure_num_cols])
                numeric_df = pd.DataFrame(numeric_data, columns=[f"num_{col}" for col in pure_num_cols])
                final_features.append(numeric_df)
                feature_info.append(f"Continuous numeric (standard scaled): {', '.join(pure_num_cols)}")

            try:
                X_all_transformed_df = pd.concat(final_features, axis=1)
                X_all_transformed_df.columns = (
                X_all_transformed_df.columns
                .str.replace(r"^catnum_", "", regex=True)
                .str.replace(r"^cattext_", "", regex=True)
                .str.replace(r"^num_", "", regex=True)      
                .str.replace(r"^tfidf_", "", regex=True))

                st.subheader("🔍 Feature Processing Summary")
                for info in feature_info:
                    st.write(info)

                st.subheader("🧾 Transformed Feature Table")
                st.dataframe(X_all_transformed_df.head())
            except Exception as e:
                st.error(f"Error combining features: {str(e)}")
                st.stop()

            # Remaining code continues unchanged...


            X_train, X_test, y_train, y_test = train_test_split(
                X_all_transformed_df, y, test_size=0.2, random_state=42)

            include_dnn = st.checkbox("Include Deep Neural Network model", value=False)

            models = {
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Naïve Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVM": SVC(probability=True),
            }
            if include_dnn:
                models["Deep Neural Network"] = "DNN"

            accuracy_scores = {}

            for name, model in models.items():
                st.subheader(f"🤖 Model: {name}")
                cached_metrics = load_metrics_if_exist(df, name)

                if cached_metrics:
                    st.write("📊 Using cached results (from previous run)")
                    st.write(f"*Accuracy:* {cached_metrics['accuracy']*100:.2f}%")
                    st.write(f"*Precision:* {cached_metrics['precision']*100:.2f}%")
                    st.write(f"*Recall:* {cached_metrics['recall']*100:.2f}%")
                    st.write(f"*F1 Score:* {cached_metrics['f1_score']*100:.2f}%")
                    accuracy_scores[name] = cached_metrics['accuracy'] * 100

                    metrics_df = pd.DataFrame({
                        'Class': label_encoder.classes_,
                        'Precision': cached_metrics['per_class_precision'],
                        'Recall': cached_metrics['per_class_recall'],
                        'F1-Score': cached_metrics['per_class_f1']
                    })
                    st.dataframe(metrics_df.style.format({
                        'Precision': '{:.2%}',
                        'Recall': '{:.2%}',
                        'F1-Score': '{:.2%}'
                    }))

                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cached_metrics['conf_matrix'], annot=True, fmt="d", cmap="Blues",
                                xticklabels=label_encoder.classes_,
                                yticklabels=label_encoder.classes_, ax=ax)
                    st.pyplot(fig)
                    continue

                if name == "Deep Neural Network":
                    y_train_cat = to_categorical(y_train)
                    y_test_cat = to_categorical(y_test)
                    input_dim = X_train.shape[1]
                    num_classes = y_train_cat.shape[1]

                    model_dnn = Sequential([
                        Dense(128, input_dim=input_dim, activation='relu'),
                        Dropout(0.3),
                        Dense(64, activation='relu'),
                        Dropout(0.3),
                        Dense(num_classes, activation='softmax')
                    ])
                    model_dnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    model_dnn.fit(X_train, y_train_cat, epochs=20, batch_size=32, verbose=0)

                    y_pred_probs = model_dnn.predict(X_test)
                    y_pred = np.argmax(y_pred_probs, axis=1)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                per_class_precision = precision_score(y_test, y_pred, average=None)
                per_class_recall = recall_score(y_test, y_pred, average=None)
                per_class_f1 = f1_score(y_test, y_pred, average=None)

                metrics_dict = {
                    'accuracy': accuracy,
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score']
                }
                save_metrics_to_db(df, name, metrics_dict, conf_matrix, per_class_precision, per_class_recall, per_class_f1)
                accuracy_scores[name] = accuracy * 100

                st.write(f"*Accuracy:* {accuracy*100:.2f}%")
                st.write(f"*Precision:* {metrics_dict['precision']*100:.2f}%")
                st.write(f"*Recall:* {metrics_dict['recall']*100:.2f}%")
                st.write(f"*F1 Score:* {metrics_dict['f1_score']*100:.2f}%")

                metrics_df = pd.DataFrame({
                    'Class': label_encoder.classes_,
                    'Precision': [report[label]['precision'] for label in label_encoder.classes_],
                    'Recall': [report[label]['recall'] for label in label_encoder.classes_],
                    'F1-Score': [report[label]['f1-score'] for label in label_encoder.classes_]
                })
                st.dataframe(metrics_df.style.format({
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}',
                    'F1-Score': '{:.2%}'
                }))

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                            xticklabels=label_encoder.classes_,
                            yticklabels=label_encoder.classes_, ax=ax)
                st.pyplot(fig)

            if accuracy_scores:
                st.subheader("📈 Model Accuracy Comparison")
                acc_df = pd.DataFrame(list(accuracy_scores.items()), columns=["Model", "Accuracy"])
                acc_df = acc_df.sort_values(by="Accuracy", ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=acc_df, x="Model", y="Accuracy", palette="viridis", ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_ylabel("Accuracy (%)")
                st.pyplot(fig)

if __name__ == "__main__":
    main()

