from src.ml.distributed import print_distribute_model

from transformers import BertModel
import torch

if __name__ == "__main__":

    dummy_input = torch.zeros((1, 1), dtype=torch.long)
    model = BertModel.from_pretrained("bert-base-uncased")
    # model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

    a = next(model.children())(dummy_input)

    print_distribute_model(model)
