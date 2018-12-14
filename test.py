from emb_path import glove_6B_50d_path
from embeddings import get_glove_embedding_matrix, sentence_to_indexes, inds_to_embedding
from gem import GEM

if __name__ == '__main__':
    TEST_SENT1 = 'i believe i can fly'
    TEST_SENT2 = 'i believe i can touch the sky'
    TEST_SENT3 = 'i think about it every night and day'
    TEST_SENT4 = 'spread my wings and fly away'

    e, v = get_glove_embedding_matrix(glove_6B_50d_path)
    embedded_sentence = inds_to_embedding(sentence_to_indexes(TEST_SENT1, v), e)
    assert embedded_sentence.shape == (e.shape[1], len(TEST_SENT1.split()))

    sentences = [TEST_SENT1, TEST_SENT2, TEST_SENT3, TEST_SENT4]
    sentences_inds = []
    for sent in sentences:
        sentences_inds.append(sentence_to_indexes(sent, v))

    gem = GEM(sentences_inds, e)
    gem_result = gem.get_sentence_embeddings(5, 20, 10)
    assert gem_result.shape == (e.shape[1], len(sentences))

    print(gem_result)
    print(gem_result.shape)
