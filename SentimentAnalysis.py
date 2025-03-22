import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load your data (adjust the file path and column name as needed)
df = pd.read_csv(r'C:\Shikha\Python\Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv')


# Suppose your text column is named 'Comments'
text_column = 'Concerns'  

# Remove rows where 'Concerns' is either NaN or an empty string
df = df.dropna(subset=[text_column])
df = df[df[text_column].str.strip() != '']


# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Compute sentiment scores and store the compound score
# Display a sample of the data with sentiment scores
df['Sentiment_Score'] = df[text_column].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else None)


print(df[[text_column, 'Sentiment_Score']].head())

# Save the DataFrame to an Excel file
output_file = r'C:\Shikha\Python\Sentiment_Analysis_Output.xlsx'
df.to_excel(output_file, index=False)

print(f"DataFrame has been saved to {output_file}")
