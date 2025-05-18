import logging
from .embeddings import EmbeddingManager 
from .encoder import EncoderRNN
from .decoder import DecoderRNN
from .autoencoder import Seq2SeqAE 

from src.config import (HIDDEN_DIM, NUM_RNN_LAYERS, DROPOUT, USE_GRU, USE_ATTENTION, ATTENTION_DIM, 
    OTHER_EMBEDDING_DIM, LEARNED_EMB_COLS, PRECOMPUTED_EMB_COLS, DIAG_EMBEDDINGS_PATH, FINETUNE_DIAG_EMBEDDINGS, MODEL_INPUT_NUM_OHE_DIM)


def build_autoencoder_from_config(logger, device) -> Seq2SeqAE:
    """ Helper function to instantiate the AE model based on config. """
    logger.info("Building AE model architecture from config...")
    # Define Embedding Manager Config
    learned_emb_config = {
        col: (vocab_size, OTHER_EMBEDDING_DIM)
        for col, vocab_size in LEARNED_EMB_COLS.items()
    }
    precomputed_emb_config = {
        col: (DIAG_EMBEDDINGS_PATH, FINETUNE_DIAG_EMBEDDINGS)
        for col in PRECOMPUTED_EMB_COLS
    }
    embedding_manager = EmbeddingManager(learned_emb_config, precomputed_emb_config, device)
    total_emb_dim = embedding_manager.get_total_embedding_dim()
    num_ohe_features = MODEL_INPUT_NUM_OHE_DIM
    logger.info(f"Determined Num OHE Features for build: {num_ohe_features}")
    encoder_input_dim = num_ohe_features + total_emb_dim

    encoder = EncoderRNN(
        num_ohe_features=num_ohe_features, embedding_manager=embedding_manager,
        hidden_dim=HIDDEN_DIM, n_layers=NUM_RNN_LAYERS, dropout=DROPOUT,
        use_gru=USE_GRU, use_attention=USE_ATTENTION
    )
    decoder = DecoderRNN(
        reconstruction_dim=encoder_input_dim, encoder_hidden_dim=HIDDEN_DIM,
        decoder_hidden_dim=HIDDEN_DIM, n_layers=NUM_RNN_LAYERS, dropout=DROPOUT,
        use_gru=USE_GRU, use_attention=USE_ATTENTION, attention_dim=ATTENTION_DIM
    )
    autoencoder = Seq2SeqAE(encoder, decoder)
    logger.info("AE model architecture built.")
    return autoencoder