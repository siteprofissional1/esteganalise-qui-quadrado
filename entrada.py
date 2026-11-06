"""Leitura de imagens para análise, preservando os bits gravados em disco."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# Modos do Pillow com canais de cor separados. "L" é cinza, "P" é paleta e "LA" e "RGBA" têm canal alfa.
_MODOS_RGB = ("RGB", "RGBA")


@dataclass
class ImagemCarregada:
    """Planos de bytes a analisar, com nomes, e avisos para mostrar ao usuário."""

    modo: str
    formato: str
    planos: dict[str, np.ndarray]
    avisos: list[str] = field(default_factory=list)


def carregar_para_analise(caminho: str) -> ImagemCarregada:
    """Lê os pixels sem converter o modo de cor.

    Converter para cinza com ``convert("L")`` combinaria os canais em uma média ponderada
    e apagaria os LSBs de uma imagem RGB. Por isso, cada canal é analisado separadamente,
    além da sequência intercalada RGBRGB..., que é a ordem usada por ferramentas que gravam
    1 LSB por canal.
    """
    with Image.open(caminho) as img:
        modo, formato = img.mode, img.format or "desconhecido"
        # Lê os bytes como estão no arquivo. Não há convert() aqui de propósito: é ele que apagaria os LSBs.
        arr = np.asarray(img)

    avisos: list[str] = []
    # O ataque age sobre LSBs de pixels. Depois de uma compressão com perdas, esses bits não são mais
    # os originais; a análise de JPEG precisaria olhar os coeficientes DCT, o que este teste não faz.
    if formato in ("JPEG", "MPO", "WEBP"):
        avisos.append(
            f"Formato {formato}: se for com perdas, a compressão teria destruído uma inserção LSB "
            "espacial, e o resultado não informa sobre esteganografia nos coeficientes DCT."
        )

    if modo == "L":
        planos = {"L": arr}
    elif modo == "LA":
        planos = {"L": arr[:, :, 0]}
        avisos.append("Canal alfa ignorado.")
    elif modo == "P":
        # Em imagem com paleta, o pixel guarda um índice para uma tabela de cores, e não a própria cor.
        planos = {"Índices da paleta": arr}
        avisos.append(
            "Imagem com paleta: o teste usa os índices, e os pares {2k, 2k+1} só têm sentido "
            "se a ferramenta de inserção usar essa mesma ordem de paleta."
        )
    elif modo in _MODOS_RGB:
        # Os três primeiros canais: descarta o alfa, se houver.
        rgb = arr[:, :, :3]
        planos = {
            "R": rgb[:, :, 0],
            "G": rgb[:, :, 1],
            "B": rgb[:, :, 2],
            "RGB intercalado": rgb,
        }
        if modo == "RGBA":
            avisos.append("Canal alfa ignorado.")
    else:
        raise ValueError(
            f"Modo de imagem não suportado: {modo}. O teste requer amostras de 8 bits "
            "(L, LA, P, RGB ou RGBA)."
        )

    return ImagemCarregada(modo, formato, {nome: np.asarray(p, dtype=np.uint8) for nome, p in planos.items()}, avisos)


def carregar_cinza(caminho: str) -> np.ndarray:
    """Lê uma imagem de cobertura e a converte para cinza antes de qualquer inserção."""
    with Image.open(caminho) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def carregar_rgb(caminho: str) -> np.ndarray:
    """Lê uma imagem de cobertura em RGB de 8 bits por canal."""
    with Image.open(caminho) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
