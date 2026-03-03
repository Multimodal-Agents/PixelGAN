import pickle, sys
path = sys.argv[1] if len(sys.argv) > 1 else "runs/pixelgan/checkpoints/step_033000/checkpoint.pkl"
with open(path, "rb") as f:
    ckpt = pickle.load(f)
blocks = [k for k in ckpt["g_params"]["synthesis"].keys() if k.startswith("block")]
image_size = 4 * (2 ** len(blocks))
print(f"Synthesis blocks: {blocks}")
print(f"=> Model was trained at --size {image_size}")
