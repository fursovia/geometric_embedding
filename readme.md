## Geometric Embedding Algorithm (GEM)

This is an implementation of Geometric Embedding Algorithm, a simple and robust non-parameterized approach for building sentence
representations. See the [paper](https://openreview.net/pdf?id=rJedbn0ctQ) for more details.

The work is done as a project for [NLA course](http://nla.skoltech.ru/) at Skoltech.

### Example

```python
from gem import SentenceEmbedder
from embeddings import get_embedding_matrix

sentences = ["We come from the land of the ice and snow",
            "From the midnight sun where the hot springs blow"]
            
embedding_matrix, vocab = get_embedding_matrix('glove.6B.300d.txt')
embedder = SentenceEmbedder(sentences, embedding_matrix, vocab)

embedded_sentences = embedder.gem(window_size=3, sigma_power=3)
```

### Data used

* [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
* [LexVec embeddings](https://github.com/alexandres/lexvec)


### Team

* [Alexey Bokhovkin](https://github.com/alexeybokhovkin)
* [Eugenia Cheskidova](https://github.com/fogside)
* [Ivan Fursov](https://github.com/fursovia)
* [Ruslan Rakhimov](https://github.com/rakhimovv)
