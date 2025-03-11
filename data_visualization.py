import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json

phishing_ds = pd.read_csv("phishing_url.csv")
df = pd.read_csv("preprocessed_data.csv")

# Visualization
# class distribution (Phishing vs. Legitimate)
sns.countplot(x=df["label"], palette=["green", "red"])
plt.title("Class Distribution (Phishing vs Legitimate URLs)")
plt.xlabel("Label (0 = Phishing, 1 = Legitimate)")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Phishing", "Legitimate"])
plt.show()

#Distribution of URL Length
sns.histplot(data=df, x="URLLength", hue="label", kde=True, bins=30, palette=["green", "red"])
plt.title(f"Distribution of {"URLLength"}")
plt.xlabel("URL Length")
plt.ylabel("Count")
plt.legend(labels=["Phishing", "Legitimate"])
plt.show()

#Distribution of Domain Length
sns.histplot(data=df, x="DomainLength", hue="label", kde=True, bins=30, palette=["green", "red"])
plt.title(f"Distribution of {"DomainLength"}")
plt.xlabel("DomainLength")
plt.ylabel("Count")
plt.legend(labels=["Phishing", "Legitimate"])
plt.show()

#Distribution of Number of Subdomains
sns.histplot(data=df, x="NoOfSubDomain", hue="label", kde=True, bins=30, palette=["green", "red"])
plt.title(f"Distribution of {"NoOfSubDomain"}")
plt.xlabel("No Of SubDomain")
plt.ylabel("Count")
plt.legend(labels=["Phishing", "Legitimate"])
plt.show()

#Distribution of URL Similarity Index
sns.histplot(data=phishing_ds, x="URLSimilarityIndex", hue="label", kde=True, bins=30, palette=["green", "red"])
plt.title(f"Distribution of {"URLSimilarityIndex"}")
plt.xlabel("URL Similarity Index")
plt.ylabel("Count")
plt.legend(labels=["Phishing", "Legitimate"])
plt.show()

#Distribution of Letter Ratio in URL
sns.histplot(data=phishing_ds, x="LetterRatioInURL", hue="label", kde=True, bins=30, palette=["green", "red"])
plt.title(f"Distribution of {"LetterRatioInURL"}")
plt.xlabel("Letter Ratio In URL")
plt.ylabel("Count")
plt.legend(labels=["Phishing", "Legitimate"])
plt.show()

#Distribution of IsHTTPS
sns.histplot(data=phishing_ds, x="IsHTTPS", hue="label", kde=True, bins=30, palette=["green", "red"])
plt.title(f"Distribution of {"IsHTTPS"}")
plt.xlabel("Is HTTPS")
plt.ylabel("Count")
plt.legend(labels=["Phishing", "Legitimate"])
plt.show()

#Correlation Heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Box Plots: Outlier Detection in Numerical Features
features_box = ["URLLength", "DomainLength", "NoOfSubDomain", "URLSimilarityIndex", "ObfuscationRatio"]
for f in features_box:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["label"], y=df[f], palette=["green", "red"])
    plt.title(f"Box Plot of {f} by Class (Phishing vs Legitimate)")
    plt.xlabel("Label (0 = Phishing, 1 = Legitimate)")
    plt.ylabel(f)
    plt.xticks(ticks=[0, 1], labels=["Phishing", "Legitimate"])
    plt.show()
    print("\n")

# Count Plot: Distribution of Categorical Features
plt.figure(figsize=(12, 6))
sns.countplot(x=phishing_ds["TLD"], hue=df["label"], palette=["green", "red"], order=phishing_ds["TLD"].value_counts().index[:10])
plt.title("Top 10 Most Common TLDs in Phishing vs. Legitimate URLs")
plt.xlabel("Top-Level Domain (TLD)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Label", labels=["Phishing", "Legitimate"])
plt.show()

#Violin Plots: Feature Distributions per Class
for feature in ["URLLength", "DomainLength", "ObfuscationRatio"]:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=df["label"], y=df[feature], palette=["green", "red"])
    plt.title(f"Violin Plot of {feature} by Class")
    plt.xlabel("Label (0 = Phishing, 1 = Legitimate)")
    plt.ylabel(feature)
    plt.xticks(ticks=[0, 1], labels=["Phishing", "Legitimate"])
    plt.show()

#HTTPS Usage in Phishing vs. Legitimate URLs
plt.figure(figsize=(6, 4))
sns.countplot(x=df["IsHTTPS"], hue=df["label"], palette=["green", "red"])
plt.title("HTTPS Usage in Phishing vs Legitimate URLs")
plt.xlabel("Uses HTTPS (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["No HTTPS", "HTTPS"])
plt.legend(title="Label", labels=["Phishing", "Legitimate"])
plt.show()

# Relationship Between Multiple Features
sns.pairplot(df, vars=["URLLength", "DomainLength", "NoOfSubDomain", "URLSimilarityIndex"], hue="label", palette=["green", "red"])
plt.show()

# Box Plots for Outliers Detection
num_features = ["NoOfPopup", "NoOfiFrame", "NoOfObfuscatedChar", "NoOfExternalRef"]
plt.figure(figsize=(12, 8))
for i, feature in enumerate(num_features, 1):
    plt.subplot(2, 2, i)  #2x2 grid layout
    sns.boxplot(x=df["label"], y=df[feature])
    plt.title(f"Box Plot: {feature} vs. Phishing Label")
plt.tight_layout() # Adjusts the spacing between subplots to prevent overlapping.
plt.show()

# Count Plot for Categorical Features
cat_features = ["HasFavicon", "HasPasswordField", "HasSubmitButton", "IsResponsive"]
plt.figure(figsize=(12, 8))
for i, feature in enumerate(cat_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=df[feature], hue=df["label"])
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

# KDE Plot for URL Character Ratios
sns.kdeplot(data=df, x="LetterRatioInURL", hue="label", fill=True, common_norm=False)
plt.show()

sns.kdeplot(data=df, x="SpacialCharRatioInURL", hue="label", fill=True, common_norm=False)
plt.show()

# Pairwise Correlation of Important Features
selected_features = ["URLLength", "DomainLength", "NoOfSubDomain", "NoOfObfuscatedChar", "NoOfPopup", "label"]
sns.pairplot(df[selected_features], hue="label")
plt.show()
