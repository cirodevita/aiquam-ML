from torch.utils.data import DataLoader
from data_provider.data_loader import UEAloader
from data_provider.uea import collate_fn


data_dict = {
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'TEST' or flag == 'VAL':
        shuffle_flag = False
    else:
        shuffle_flag = True

    batch_size = args.batch_size
    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        flag=flag,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    return data_set, data_loader