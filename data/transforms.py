import torchvision.transforms as transforms

def transform_train(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    img_tensor = transform(img)
    return img_tensor

def transform_val(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    img_tensor = transform(img)
    return img_tensor
