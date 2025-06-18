import torch

from utils.masking import create_mask


def generate_predictions(model, src, device="cuda"):
    batch_size = src.shape[0]
    N = src.shape[-1]
    output = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)

    for _ in range(N - 1):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, output, pad_idx=4, device=device
        )

        logits = model(
            src,
            output,
            tgt_mask=tgt_mask,
        )

        idx = logits[:, -1].argmax(dim=-1).view(-1, 1)
        output = torch.cat((output, idx), dim=-1)

    return output
