from wb_data import WaterBirdsDataset, get_transform_cub, get_loader, log_data


def load_data(basedir, batch_size, target_resolution=(224, 224)):
    train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=True)
    val_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=True)
    test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False)

    train_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='train',
        transform=train_transform
        )
    val_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='val',
        transform=val_transform
        )
    test_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='test',
        transform=test_transform
        )

    train_dataloader = get_loader(
        data=train_dataset,
        train=True,
        reweight_groups=None,
        reweight_classes=None,
        reweight_places=None,
        batch_size=batch_size,
        )
    val_dataloader = get_loader(
        data=val_dataset,
        train=False,
        reweight_groups=None,
        reweight_classes=None,
        reweight_places=None,
        batch_size=batch_size,
        )
    test_dataloader = get_loader(
        data=test_dataset,
        train=False,
        reweight_groups=None,
        reweight_classes=None,
        reweight_places=None,
        batch_size=batch_size,
        )
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    # for testing only
    train_dataloader, val_dataloader, test_dataloader = load_data('/kaggle/working/waterbird_complete95_forest2water2', batch_size=1)
    print(f'len train:\t{len(train_dataloader)}')
    print(f'len validation:\t{len(val_dataloader)}')
    print(f'len test:\t{len(test_dataloader)}')
    for img, y, g, p in train_dataloader:
        print(img.shape, y.shape, g.shape, p.shape, sep='\t')
        print(y, g, p, sep='\t')
    exit()