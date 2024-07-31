importar  pandas  como  pd 
importar  matplotlib.pyplot  como  plt 
importar  seaborn  como  sns 
de  wordcloud  importar  WordCloud 
de  textblob  importar  TextBlob 
importar  nltk 
de  nltk.corpus  importar  stopwords 
de  sklearn.feature_extraction.text  importar  CountVectorizer

#-se de que o nltk certifique-se de baixar 
nltk . baixar ( 'palavras irrelevantes' )

# URL do arquivo CSV no Google Drive 
url  =  'https://drive.google.com/uc?export=download&id=1_TPCGMVyhOhw0Jyl7ICIEPG3OnNta-EV'

# Ler o arquivo CSV diretamente do Google Drive 
df  =  pd . read_csv ( url )

# Análise Exploratória de Dados (AED) 
print ( "Primeiras linhas do DataFrame:" ) 
print ( df . head ())

print ( " \n Informações do DataFrame:" ) print 
( df.info ( ) )

print ( " \n Estatísticas descritivas:" ) 
print ( df .describe ( ))

# Pré-processamento de dados 
df [ 'paper_version_release_date' ]  =  pd . to_datetime ( df [ 'paper_version_release_date' ]) 
df [ 'paper_subject' ]  =  df [ 'paper_subject' ] . astype ( str ) 
df [ 'paper_abstract' ]  =  df [ 'paper_abstract' ] . astype ( str )

# Identificação de Tendências 
df [ 'year' ]  =  df [ 'paper_version_release_date' ] . dt . year 
yearly_counts  =  df [ 'year' ] . value_counts () . sort_index ()

# Gráfico 1: Número de Publicações por Ano (Histograma) 
plt . figura ( figsize = ( 10 , 6 )) 
sns . histplot ( df [ 'ano' ],  bins = 10 ,  kde = False ) 
plt . title ( 'Número de Publicações por Ano' ) 
plt . xlabel ( 'Ano' ) 
plt . ylabel ( 'Número de Publicações' ) 
plt . grade ( Verdadeiro ) 
plt . mostrar ()

# Principais Tópicos 
vectorizer  =  CountVectorizer ( stop_words = 'english' ,  max_features = 10 ) 
X  =  vectorizer . fit_transform ( df [ 'paper_abstract' ]) 
top_words  =  vectorizer . get_feature_names_out ()

word_counts  =  X . toarray () . sum ( axis = 0 ) 
word_freq  =  dict ( zip ( top_words ,  word_counts ))

# Gráfico 2: Principais Tópicos em Resumos (Gráfico de Barras) 
plt . figura ( figsize = ( 10 , 6 )) 
sns . barplot ( x = lista ( word_freq . valores ()),  y = lista ( word_freq . chaves ())) 
plt . título ( 'Principais Tópicos em Resumos' ) 
plt . xlabel ( 'Frequência' ) 
plt . ylabel ( 'Tópicos' ) 
plt . grade ( Verdadeiro ) 
plt . mostrar ()

# Análise de Sentimento
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

df['clean_abstract'] = df['paper_abstract'].apply(preprocess_text)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['sentiment'] = df['clean_abstract'].apply(analyze_sentiment)

# Gráfico 3: Distribuição de Sentimentos dos Abstracts (Gráfico de Dispersão)
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='year', y='sentiment', hue='paper_subject')
plt.title('Distribuição de Sentimentos dos Abstracts por Ano e Assunto')
plt.xlabel('Ano')
plt.ylabel('Polaridade do Sentimento')
plt.grid(True)
plt.legend(title='Assunto')
plt.show()

# Relatório Completo
report = """
Relatório de Análise de Dados sobre Ética em Inteligência Artificial

1. Introdução
Este relatório apresenta uma análise dos dados sobre ética em Inteligência Artificial (IA) usando um conjunto de dados de artigos disponíveis. O objetivo é fornecer insights relevantes para ajudar os clientes a entender as tendências, os principais temas e as preocupações relacionadas à ética em IA com base nos artigos disponíveis.

2. Análise Exploratória de Dados (AED)
O conjunto de dados contém informações detalhadas sobre artigos publicados, incluindo referências, links, títulos, autores, assuntos, resumos, datas de lançamento e mais. Os dados foram pré-processados para garantir a consistência e a correta tipagem das variáveis.

3. Identificação de Tendências
A análise das tendências ao longo do tempo mostrou um aumento no número de publicações
"""

print(report)
[nltk_data] Downloading package stopwords to
[nltk_data]     /home/d1a06a60-c542-42b2-bdb5-
[nltk_data]     7ee2a6636788/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Primeiras linhas do DataFrame:
  arXiv_reference                  paper_arXiv_link  \
0      2205.03468  https://arxiv.org/abs/2205.03468   
1      2001.00081  https://arxiv.org/abs/2001.00081   
2      2109.07906  https://arxiv.org/abs/2109.07906   
3      2206.07635  https://arxiv.org/abs/2206.07635   
4      2206.03225  https://arxiv.org/abs/2206.03225   

                                         paper_title  \
0                    The AI Index 2022 Annual Report   
1  Exciting, Useful, Worrying, Futuristic:\r\nPub...   
2  Ethics of AI: A Systematic Literature Review o...   
3  AI Ethics Issues in Real World: Evidence from ...   
4  The Different Faces of AI Ethics Across the Wo...   

                                       paper_authors  \
0  Daniel Zhang, Nestor Maslej, Erik Brynjolfsson...   
1  Patrick Gage Kelley, Yongwei Yang, Courtney He...   
2  Arif Ali Khan, Sher Badshah, Peng Liang, Bilal...   
3                           Mengyi Wei, Zhixuan Zhou   
4              Lionel Nganyewou Tidjon, Foutse Khomh   

                                       paper_subject  \
0                    Artificial Intelligence (cs.AI)   
1  Computers and Society (cs.CY); Artificial Inte...   
2  Computers and Society (cs.CY); Artificial Inte...   
3  Artificial Intelligence (cs.AI); Computers and...   
4  Computers and Society (cs.CY); Artificial Inte...   

                                      paper_abstract  \
0  Welcome to the fifth edition of the AI Index R...   
1  As the influence and use of artificial intelli...   
2  Ethics in AI becomes a global topic of interes...   
3  With the powerful performance of Artificial In...   
4  Artificial Intelligence (AI) is transforming o...   

                              paper_doi_link  paper_number_of_pages  \
0  https://doi.org/10.48550/arXiv.2205.03468                  230.0   
1  https://doi.org/10.48550/arXiv.2001.00081                   12.0   
2  https://doi.org/10.48550/arXiv.2109.07906                   21.0   
3  https://doi.org/10.48550/arXiv.2206.07635                    9.0   
4  https://doi.org/10.48550/arXiv.2206.03225                   20.0   

  paper_version_release_date paper_ondatabase_since paper_ondatabase_latest  
0                 2022-05-02             2023-07-03                     NaN  
1                 2021-05-18             2023-07-03                     NaN  
2                 2021-09-12             2023-07-03                     NaN  
3                 2022-08-18             2023-07-03                     NaN  
4                 2022-05-12             2023-07-03                     NaN  

Informações do DataFrame:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 196 entries, 0 to 195
Data columns (total 11 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   arXiv_reference             196 non-null    object 
 1   paper_arXiv_link            196 non-null    object 
 2   paper_title                 196 non-null    object 
 3   paper_authors               196 non-null    object 
 4   paper_subject               195 non-null    object 
 5   paper_abstract              195 non-null    object 
 6   paper_doi_link              195 non-null    object 
 7   paper_number_of_pages       195 non-null    float64
 8   paper_version_release_date  196 non-null    object 
 9   paper_ondatabase_since      196 non-null    object 
 10  paper_ondatabase_latest     23 non-null     object 
dtypes: float64(1), object(10)
memory usage: 17.0+ KB
None

Estatísticas descritivas:
       paper_number_of_pages
count             195.000000
mean               22.738462
std                23.992016
min                 5.000000
25%                11.000000
50%                16.000000
75%                26.000000
max               230.000000
No description has been provided for this image
No description has been provided for this image
/opt/conda/envs/anaconda-panel-2023.05-py310/lib/python3.11/site-packages/IPython/core/pylabtools.py:152: UserWarning: Glyph 9 (	) missing from current font.
  fig.canvas.print_figure(bytes_io, **kw)
No description has been provided for this image
Relatório de Análise de Dados sobre Ética em Inteligência Artificial

1. Introdução
Este relatório apresenta uma análise dos dados sobre ética em Inteligência Artificial (IA) usando um conjunto de dados de artigos disponíveis. O objetivo é fornecer insights relevantes para ajudar os clientes a entender as tendências, os principais temas e as preocupações relacionadas à ética em IA com base nos artigos disponíveis.

2. Análise Exploratória de Dados (AED)
O conjunto de dados contém informações detalhadas sobre artigos publicados, incluindo referências, links, títulos, autores, assuntos, resumos, datas de lançamento e mais. Os dados foram pré-processados para garantir a consistência e a correta tipagem das variáveis.

3. Identificação de Tendências
A análise das tendências ao longo do tempo mostrou um aumento no número de publicações

import matplotlib.pyplot as plt
import seaborn as sns

# Dados simulados para exemplo
anos = [2019, 2020, 2021, 2022, 2023]
assuntos = ['Ética', 'Privacidade', 'Transparência']
sentimentos_positivos = [10, 15, 20, 25, 30]
sentimentos_neutros = [5, 10, 12, 15, 20]
sentimentos_negativos = [2, 5, 8, 10, 12]

# Criar o gráfico de barras empilhadas
plt.figure(figsize=(10, 6))
sns.set_palette("viridis")
plt.bar(anos, sentimentos_positivos, label='Positivo', color='lightgreen')
plt.bar(anos, sentimentos_neutros, bottom=sentimentos_positivos, label='Neutro', color='lightblue')
plt.bar(anos, sentimentos_negativos, bottom=[i+j for i,j in zip(sentimentos_positivos, sentimentos_neutros)], 
        label='Negativo', color='salmon')

plt.xlabel('Ano')
plt.ylabel('Número de Artigos')
plt.title('Distribuição de Sentimentos dos Abstracts por Ano e Assunto')
plt.xticks(anos)
plt.legend()
plt.tight_layout()

# Mostrar o gráfico
plt.show()
No description has been provided for this image
 
