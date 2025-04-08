"""
@package raw

Funciones para lectura y escritura de archivos con formato .raw
"""

import struct
import numpy as np
from PySide6.QtCore import QBuffer, QSharedMemory
import gzip
import os
import lz4.block  # type: ignore




TYPE_DICT = {
    b"FL4B": ("f", 1, 4),
    b"F24B": ("f", 2, 4),
    b"F34B": ("f", 3, 4),
    b"F44B": ("f", 4, 4),
    b"SI4B": ("i", 1, 4),
    b"UI1B": ("B", 1, 1),
    b"U31B": ("B", 3, 1),
    b"UI2B": ("H", 1, 2),
    b"FL8B": ("d", 1, 8),
    b"F28B": ("d", 2, 8),
    b"F48B": ("d", 4, 8),

}


def read_img_raw_SharedMemory(sharedMemory: QSharedMemory) -> np.ndarray:
    """
    Lee un archivo .raw de un bloque de memoria compartido. Ejemplo de uso:
    sharedMemory = QSharedMemory("key")
    sharedMemory.attach()
    img=read_img_raw_SharedMemory(sharedMemory)

    Args
    ---
        - sharedMemory : Objeto QSharedMemory

    Returns
    ---
        - numpy.array
    """
    sharedMemory.lock()
    fid = QBuffer()
    fid.setData(sharedMemory.constData())  # type: ignore
    fid.open(QBuffer.ReadOnly)  # type: ignore
    header = fid.read(8)
    if header == b"IMG_INFO":
        nr = struct.unpack("i", fid.read(4))[0]  # type: ignore
        nc = struct.unpack("i", fid.read(4))[0]  # type: ignore
        nch = 1
        type = fid.read(4)
        if type == b"FL4B":
            bytes = 4
            typem = "f"
        elif type == b"F24B":
            bytes = 4 * 2
            typem = "f"
            nch = 2
        elif type == b"F34B":
            bytes = 4 * 3
            typem = "f"
            nch = 3
        elif type == b"SI4B":
            bytes = 4
            typem = "i"
        elif type == b"UI1B":
            bytes = 1
            typem = "c"
        elif type == b"UI2B":
            bytes = 2
            typem = "H"
        elif type == b"FL8B":
            bytes = 8
            typem = "d"
        elif type == b"F28B":
            bytes = 8 * 2
            typem = "d"
            nch = 2
        else:
            fid.close
            sharedMemory.unlock()
            return np.zeros([])
        try:
            data = fid.read(bytes * nr * nc)
            d = np.frombuffer(data, dtype=typem)  # type: ignore
            fid.close()
            sharedMemory.unlock()
            img = np.reshape(d, (nr, nc, nch), order="C").copy()
            return np.squeeze(img)
        except Exception as e:
            DSI_LOG_ERROR(f"Error leyendo espacio de memoria. Código de error: {e}")
            return np.zeros([])
    else:
        DSI_LOG_ERROR(f"Can't read {sharedMemory.key()}")
        return np.zeros([])


def read_img_raw(filename: str) -> np.ndarray:
    """
    Función que permite leer un archivo .raw.

    Args
    ---
        - filename : Dirección del archivo a leer.

    Returns
    ---
        - numpy.array.
    """
    if not os.path.isfile(filename):
        if os.path.isfile(filename + ".gz"):
            filename += ".gz"
        elif os.path.isfile(filename + ".lz4"):
            filename += ".lz4"
    if ".lz4" in filename:
        return read_img_raw_lz4(filename)
    if ".gz" in filename:
        fid = gzip.open(filename, "rb")
    else:
        fid = open(filename, "rb")  # type: ignore
    end = False
    final_img = np.array([])
    while not end:
        header = fid.read(8)
        if len(header) < 8:
            end = True
        if not end:
            if header != b"IMG_INFO":
                raise ValueError("Invalid file format")
            nr, nc = struct.unpack("i i", fid.read(8))

            type = fid.read(4)
            if type not in TYPE_DICT:
                DSI_LOG_ERROR("Invalid file format found {!r}".format(type))
                raise ValueError("Invalid file format")
            typem, nch, bytes = TYPE_DICT[type]
            img_data = np.frombuffer(fid.read(nr * nc * nch * bytes), dtype=typem)
            img_data = img_data.reshape((nr, nc, nch))
            if final_img.size == 0:
                final_img = np.copy(img_data)
                final_img.setflags(write=True)
            else:
                final_img = np.concatenate((final_img, img_data), axis=-1)
    return final_img


def python_type_to_short_type(typem: str) -> bytes:
    types = b""
    if typem == "float32":
        types = b"FL4B"
    elif typem == "float32C2":
        types = b"F24B"
    elif typem == "float32C3":
        types = b"F34B"
    elif typem == "double":
        types = b"FL8B"
    elif typem == "int32":
        types = b"SI4B"
    elif typem == "uchar":
        types = b"UI1B"
    elif typem == "ushort":
        types = b"UI2B"
    elif typem == "doubleC4":
        types = b"F48B"
    else:
        raise ValueError(f"type {typem} not found.")
    return types


def write_img_raw(img: np.ndarray, filename: str, type: str, mode: str) -> bool:
    """
    Función que permite escribir un archivo .raw.

    Args
    ---
        - img : Imagen en formato np.array.
        - filename : Dirección del archivo a escribir.
        - type : formato de imagen('float32','double','int32','uchar','ushort')
        - mode : Modo de acceso al archivo. 'wb' para escribir binario.

    Returns
        Bool
    ---
    """
    try:
        s = img.shape
        if ".gz" in filename:
            fid = gzip.open(filename, mode)  # type: ignore
        else:
            fid = open(filename, mode)  # type: ignore
        fid.write(b"IMG_INFO")  # type: ignore
        np.array(s[0], dtype=np.int32).tofile(fid)
        np.array(s[1], dtype=np.int32).tofile(fid)
        fid.write(python_type_to_short_type(type))  # type: ignore

        typem, _, _ = TYPE_DICT[python_type_to_short_type(type)]

        np.array(img, dtype=typem).tofile(fid)
        fid.close()
    except Exception as e:
        DSI_LOG_ERROR(e)
        return False
    return True


def read_img_raw_lz4(filename: str) -> np.ndarray:
    """
    Función que permite leer un archivo .raw.

    Args
    ---
        - filename : Dirección del archivo a leer.

    Returns
    ---
        - numpy.array.
    """
    with open(filename, "rb") as f:
        compressed_data = f.read()
    try:
        decompressed_data = lz4.block.decompress(compressed_data, return_bytearray=True)
    except lz4.block.LZ4BlockError as e:
        raise ValueError(
            "Decompression failed: corrupt input or insufficient space in destination buffer."
        ) from e

    index = 0
    header = decompressed_data[index : index + 8]
    index = index + 8
    if header != b"IMG_INFO":
        raise ValueError("Invalid file format")

    nr, nc = struct.unpack("i i", decompressed_data[index : index + 8])
    index = index + 8
    type = bytes(decompressed_data[index : index + 4])

    index = index + 4

    if type not in TYPE_DICT.keys():
        DSI_LOG_ERROR("Invalid file format found {!r}".format(type))
        raise ValueError("Invalid file format")

    typem, nch, _ = TYPE_DICT[type]

    img_data = np.frombuffer(decompressed_data, dtype=typem, offset=index)
    img_data = img_data.reshape((nr, nc, nch))
    return img_data
