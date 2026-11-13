from utils.brinkman_amitex import load_stokesbrinkman

root_dir = r"/mnt/data2/yc2634/ML/StokesBrinkman/R64"
train_loader, test_loaders, data_processor = load_stokesbrinkman(
    root_dir=root_dir,
    n_train=1,
    n_tests=1,
    batch_size=1,
    test_batch_sizes=[1],
    test_resolutions=[64],
    train_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=True,
    encode_output=True,
    loading_cond='[1,0]',
)
