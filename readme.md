[Zero-training Sentence Embedding via Orthogonal Basis](https://arxiv.org/abs/1810.00438) paper implementation

* [Project proposal](https://docs.google.com/document/d/1aok_e_UXDNRH9HvOZ6grawrZI4OjVrHfEJFuca_t4ng/edit?usp=sharing)
* [Presentation](https://docs.google.com/presentation/d/1EEmoU7C_RjBmJJD3YF3RLm2AsxTZKvtVHYFWGsX3vjo/edit?usp=sharing)
* [Report](https://docs.google.com/document/d/1XsH6srwFwoKXkvMspJmy-bHMXq-fuqNrnXJcEv2fFqQ/edit?usp=sharing)


Data used:

* [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
* [LexVec embeddings](https://github.com/alexandres/lexvec)
* [STS data](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)
* [IMDB/Reuters](https://keras.io/datasets/)

| Model                      | Train |  Dev  | Test |
|----------------------------|-------|-------|------|
| Universal Sentence Encoder | 69.7  | ?     | ?    |
| GEM+Glove                  | ?     | 79.9  | 64.9 |
| GEM+LexVec                 | ?     | 76.9  | 63.3 |
| Lexvec                     | ?     | 66.6  | 38.8 |
| Glove                      | ?     | 51.8  | 27.4 |
| ELMO                       | ?     | ?     | ?    |

## Глобальные задачи

* [ ] Преза
* [ ] Репорт
* [ ] Реализация алгоритма
* [ ] Сравнение с другими методами на STS, text classification (докидываем softmax)
* [ ] (?) какая-нибудь легкая модернизация алгоритма, которая добавляет качество
* [ ] Всякие графики/таблицы по скорости работы алгоритма, его качества на разных задачах


## Подзадачи

### Работа с данными

* подкрутить nltk для токенизации (нужно сделать как в статье)
* подгрузка и обработка STS/IMDB/Reuters datasets


### Аналитика

* Реализовать функции, которые считают метрики (пирсона для STS)
* Ноутбуки, в которых показывается, как запускать модель
* Ноутбуки, в которых графики и таблицы

### Модель

* Реализовать GEM (модель из статьи)
* Релиазовать модели, с которыми мы будем сравнивать 
(самое простое простое -- это усреднение эмбеддингов с помощью tf-idf)
* найти простые реализации моделей, которые можно бесшовно запустить
(модели из gensim [sent2vec, doc2vec], fasttext, tfhub, smth else?) 

Вот реализация какого-то крутого метода: https://github.com/kawine/usif

Еще нужно посмотреть, с чем ребята сами себя сравнивали в статье.

### Преза-репорт

Здесь нужно хорошо расписать всю математику, знать доказательство всего подряд.

* Предыстория (что такое эмбеддинги, какие есть методы представления текста)
* Рассказать про semantic textual similarity (что такое)
* Математически вывести формулы, объяснить интуицую
* Сравнить модель с другими моделями: скорость и качество
* Рассказать кратко про другие unsupervised подходы

