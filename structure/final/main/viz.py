import pandas as pd
import numpy as np

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Counter
from collections import Counter

# Wordclouds
from wordcloud import WordCloud



def plot_numeric_histograms(df, n_cols = 3):
    
    columns = df.select_dtypes(include='number').columns
    
    n_rows = (len(columns) // n_cols) + (len(columns) % n_cols > 0)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axes
    axes = axes.flatten()

    # Plot histograms for each column 
    for i, col in enumerate(columns):
        ax = axes[i]
        ax.hist(df[col].dropna(), edgecolor='black', color='orange')
        ax.set_title(f'Histogram of {col}', fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    # Remove empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Plot Histograms
def plot_histogram(df, column, rotation = 45):

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], discrete=True, color='orange', kde=False)
    plt.title(f'Histogram of {column}', fontsize=14)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=rotation)
    plt.show()



def plot_pairwise_relationship(df):

    # Select numerical columns
    num_columns = df.select_dtypes(include=['number']).columns

    # Create a pairplot with only the lower triangle
    g = sns.pairplot(df[num_columns], kind='scatter', 
                 hue=None, plot_kws={'s': 10, 'color': 'orange'}, 
                 corner=True) 
    
    # Update the diagonal plots to be orange
    for ax in g.diag_axes:
        for patch in ax.patches:
            patch.set_facecolor('orange')

        
    plt.show()


def plot_crosstab(df, column1, column2, annot_kws={"rotation": 45}):

    # Create the crosstab
    crosstab = pd.crosstab(df[column1], df[column2])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap='Oranges', annot_kws=annot_kws)
    plt.title(f'{column1} vs {column2}')
    plt.show()


def plot_categ_cont(df, categorical, continuous, n_cols=3):

    # Calculate the number of rows needed
    n_cats = len(categorical)
    n_cont = len(continuous)
    n_rows = int(np.ceil(n_cats / n_cols))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6, n_rows * 6))
    
    # Flatten axes
    axes = axes.flatten()

    plot_idx = 0
    for i, cat in enumerate(categorical):
        for j, cont in enumerate(continuous):
            if plot_idx < len(axes):
                sns.boxplot(x=cat, y=cont, data=df, palette='Oranges', ax=axes[plot_idx])
                axes[plot_idx].set_title(f'{cat} vs {cont}')
                axes[plot_idx].tick_params(axis='x', rotation=45)
                plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_cat_vs_num(df, categorical_column, numerical_column, plot_type="box"):
    plt.figure(figsize=(10, 6))
    palette = 'Oranges'
    
    # Sort Categores
    category_order = sorted(df[categorical_column].astype(str).unique())

    # Plot Boxplot
    if plot_type == "box":
        sns.boxplot(data=df, x=categorical_column, y=numerical_column, 
                    palette=palette, order=category_order)
        
    # Plot Bar chart
    elif plot_type == "bar":
        sns.barplot(data=df, x=categorical_column, y=numerical_column, estimator="mean", 
                    palette=palette, order=category_order)
        
    plt.title(f'{categorical_column} vs {numerical_column}')
    plt.xticks(rotation = 45)
    plt.show()




def generate_wordcloud(df, column_name, title='Word Cloud', width=800, height=400, colormap='Oranges', background_color='white', max_words=None):
    # Combine text into a single string
    all_text = " ".join(df[column_name].astype(str).tolist()).lower()
    
    # Tokenize the text
    words = all_text.split()

    # Count the word frequencies
    word_counts = Counter(words)

    # Generate the word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words
    ).generate_from_frequencies(word_counts)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

def plot_model_metrics(df, metrics, color="orange"):
    df = df.reset_index().rename(columns={"index": "Model"})
    
    for metric in metrics:
        # Sort column order
        sorted_df = df.sort_values(by=metric, ascending=True)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Model", y=metric, data=sorted_df, color=color)

        plt.title(f"{metric.replace('_', ' ').title()}", fontsize=16)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()  
        plt.show()


def plot_metrics_kfold(df):
    """
    Input: dataframe
    Output: metrics plot
    """
    metrics = df.index 
    models = df.columns 

    # For each metric
    for metric in metrics:
        values = [float(value.split("+/-")[0]) for value in df.loc[metric]]
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.bar(models, values, color="orange")
        plt.title(metric, fontsize=16)
        plt.ylabel("Value", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.tight_layout()
        plt.show()