import os
import mxnet as mx
import cv2
import lmdb
import msgpack


def rec2image(path_imgrec, path_imgidx, output):
    """ 
    Taken from deepinsight/insightface
    https://github.com/deepinsight/insightface/blob/master/recognition/data/rec2image.py
    """
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    os.makedirs(output, exist_ok=True)
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag > 0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    pp = 0
    for identity in seq_identity:
        id_dir = os.path.join(output, str(identity))
        os.makedirs(id_dir)
        pp += 1
        if pp % 10 == 0:
            print('processing id', pp)
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        imgid = 0
        for _idx in range(int(header.label[0]), int(header.label[1])):
            s = imgrec.read_idx(_idx)
            _header, _img = mx.recordio.unpack(s)
            _img = mx.image.imdecode(_img).asnumpy()[:, :, ::-1]  # to bgr
            image_path = os.path.join(id_dir, "%d.jpg" % imgid)
            cv2.imwrite(image_path, _img)
            imgid += 1


def image2lmdb(input_path, output_path, write_frequency=5000):
    """Convert ImageFolder to LMDB dataset
    key = index
    value = MessagePack serialized (image, label) pair
    number of identities stored in __n_classes__
    """
    db = lmdb.open(output_path, map_size=2**41, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    idx = 0
    for identity, folder in enumerate(os.scandir(input_path)):
        if folder.is_file():
            continue
        for file in os.scandir(folder):
            if file.is_dir():
                continue
            with open(file, 'rb') as f:
                image = f.read()
            txn.put(str(idx).encode('ascii'), msgpack.packb((image, identity)))
            idx += 1
            if idx % write_frequency == 0:
                print(idx)
                txn.commit()
                txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__n_classes__', str(identity + 1).encode('ascii'))

    db.sync()
    db.close()
