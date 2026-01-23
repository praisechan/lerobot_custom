import torch
from pi0_infer import Pi0Inference
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_views', type=int, default=2, help='Number of views')
    parser.add_argument('--prompt_len', type=int, default=0, help='Prompt length')
    parser.add_argument('--chunk_size', type=int, default=63, help='Chunk size')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    args = parser.parse_args()

    if args.checkpoint_dir:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
    else:
        checkpoint = {'language_embeds': torch.empty(args.prompt_len, 2048, dtype=torch.bfloat16)}

    infer = Pi0Inference(checkpoint, num_views=args.num_views, chunk_size=args.chunk_size)

    input_image = torch.empty(args.num_views, 224, 224, 3, dtype=torch.bfloat16).cuda()
    input_state = torch.empty(32, dtype=torch.bfloat16).cuda()
    input_noise = torch.empty(args.chunk_size, 32, dtype=torch.bfloat16).cuda()

    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("Concurrent Encoder Decoder")
        _ = infer.forward(input_image, input_state, input_noise, concurrent=True)
        torch.cuda.nvtx.range_pop()            
        torch.cuda.synchronize()

    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("Concurrent Encoder Decoder")
        _ = infer.forward(input_image, input_state, input_noise, concurrent=False)
        torch.cuda.nvtx.range_pop()            
        torch.cuda.synchronize()

    torch.cuda.synchronize()
