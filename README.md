# Python Data Science & Machine Learning Toolkit

### 1. Data Handling & Manipulation

**pandas** (`pd`)
- Reading and writing data: `read_csv`, `to_csv`, `read_excel`
- Data cleaning: `dropna`, `fillna`, `replace`
- Data manipulation: `groupby`, `merge`, `pivot_table`

**numpy** (`np`)
- Numerical operations: `np.sqrt`, `np.exp`, `np.log`
- Array manipulations: `np.reshape`, `np.concatenate`
- Random number generation: `np.random.rand`, `np.random.normal`

---

### 2. Data Visualization

**matplotlib.pyplot** (`plt`)
- Basic plots: `plot`, `scatter`, `hist`, `bar`
- Customization: `xlabel`, `ylabel`, `title`, `legend`

**seaborn** (`sns`)
- Statistical plots: `sns.heatmap`, `sns.pairplot`, `sns.boxplot`
- Categorical plots: `sns.barplot`, `sns.violinplot`

**plotly** (`plotly.express` / `plotly.graph_objects`)
- Interactive visualizations: `px.scatter`, `px.line`, `px.bar`
- 3D plots and dashboards: `go.Figure`, `update_layout`

---

### 3. Statistical Analysis & Hypothesis Testing

**scipy.stats** (`stats`)
- Normality tests: `shapiro`, `kstest`
- T-tests and ANOVA: `ttest_ind`, `f_oneway`
- Probability distributions: `norm.pdf`, `expon.rvs`

**statsmodels.api** (`sm`)
- Ordinary Least Squares (OLS) regression: `OLS`
- Time series analysis: `tsa`
- Generalized Linear Models: `GLM`

---

### 4. Machine Learning & Model Training

**sklearn.linear_model.LinearRegression**
- Linear regression: `LinearRegression().fit(X, y)`
- Model coefficients: `model.coef_`, `model.intercept_`

**sklearn.model_selection.train_test_split**
- Data splitting: `train_test_split(X, y, test_size=0.2, random_state=42)`

**scikit-learn (`sklearn`)**
- Classification: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
- Clustering: `KMeans`, `DBSCAN`
- Dimensionality Reduction: `PCA`, `TSNE`

---

### 5. Deep Learning

**tensorflow** (`tf`)
- Building neural networks: `tf.keras.models`, `tf.layers`
- Training models: `tf.fit`, `tf.evaluate`

**pytorch** (`torch`)
- Deep learning: `torch.nn`, `torch.optim`, `torch.autograd`
- GPU acceleration: `torch.cuda`

---

### 6. Natural Language Processing (NLP)

**nltk**
- Text preprocessing: `word_tokenize`, `stopwords`, `stemmer`
- POS tagging and Named Entity Recognition: `ne_chunk`, `pos_tag`

**spacy**
- Efficient NLP pipelines: `nlp()`, `doc.ents`, `doc.noun_chunks`

---

### 7. Web Scraping

**beautifulsoup4** (`bs4`)
- Parsing HTML: `BeautifulSoup(html, "html.parser")`
- Extracting data from web pages: `find`, `find_all`

**scrapy**
- Web crawling and automation: `scrapy.Spider`, `scrapy.Request`

---

### 8. Working with APIs

**requests**
- Making HTTP requests: `requests.get`, `requests.post`
- Handling responses: `response.json()`, `response.status_code`

---

### 9. Data Storage & Databases

**sqlalchemy**
- Database connections: `create_engine`, `sessionmaker`
- Query execution: `engine.execute`, `session.query`

**sqlite3**
- Handling SQLite databases: `sqlite3.connect`, `cursor.execute`
