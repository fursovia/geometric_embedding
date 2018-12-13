from emb_path import glove_6B_50d_path
from embeddings import get_embedding_matrix, embedding_lookup
from gem import GEM

if __name__ == '__main__':
    TEST_SENT1 = 'i believe i can fly'
    TEST_SENT2 = 'i believe i can touch the sky'
    TEST_SENT3 = 'i think about it every night and day'
    TEST_SENT4 = 'spread my wings and fly away'

    e, v = get_embedding_matrix(glove_6B_50d_path)
    embedded_sentence = embedding_lookup(TEST_SENT1, e, v)
    assert embedded_sentence.shape == (e.shape[1], len(TEST_SENT1.split()))

    sentences = [TEST_SENT1, TEST_SENT2, TEST_SENT3, TEST_SENT4]
    gem = GEM(glove_6B_50d_path, sentences)
    gem_result = gem.get_sentence_embeddings(5, 20, 10)
    assert gem_result.shape == (e.shape[1], len(sentences))
    print(gem_result)
    print(gem_result.shape)
