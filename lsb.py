"""Inserção esteganográfica LSB, sequencial e pseudoaleatória, e extração de mensagens.

O QUE É LSB
    LSB (Least Significant Bit) é o bit menos significativo do byte. Trocá-lo muda o valor do
    pixel em no máximo 1 de 256 níveis, o que é invisível a olho nu.

        Pixel 154 = 10011010. Gravar o bit 1 -> 10011011 = 155.
        Conta usada: (154 & 0xFE) | 1
        - "& 0xFE" (11111110) zera o último bit e preserva os outros sete;
        - "| bit" escreve o bit da mensagem nessa posição.

OS DOIS MÉTODOS
    - Sequencial: usa os primeiros pixels. É simples, mas concentra a alteração no início.
    - Pseudoaleatório: usa pixels sorteados a partir de uma chave. Espalha a alteração, e quem
      tem a chave consegue refazer o mesmo sorteio para ler a mensagem.

DOIS MODOS DE USO
    - ``inserir_lsb_*``: grava bits aleatórios em uma fração da imagem. Simula uma mensagem
      cifrada, cujos bits são praticamente uniformes. É o que o experimento usa.
    - ``inserir_mensagem`` / ``extrair_mensagem``: grava bytes reais, com um cabeçalho de 4 bytes
      indicando o tamanho, e permitem recuperá-los. É o que a demonstração usa.

    A "taxa de payload" é a fração de pixels usados. Em média, só metade desses pixels muda de
    valor, pois o bit gravado já coincide com o LSB original na outra metade.
"""

import numpy as np

METODOS = ("sequencial", "pseudoaleatorio")

# Cabeçalho da mensagem: 4 bytes guardam o tamanho, então o leitor sabe onde ela termina.
_BYTES_CABECALHO = 4


def _validar_imagem(imagem: np.ndarray) -> None:
    """Exige uint8 para evitar conversões silenciosas que corromperiam os pixels."""
    if not isinstance(imagem, np.ndarray) or imagem.dtype != np.uint8:
        tipo = getattr(imagem, "dtype", type(imagem))
        raise TypeError(f"A imagem deve ser um ndarray uint8; recebido: {tipo}.")


def _bits_por_taxa(total: int, taxa_payload: float) -> int:
    """Converte a taxa de payload em bits: cada pixel (ou canal) guarda exatamente 1 bit."""
    if not 0.0 <= taxa_payload <= 1.0:
        raise ValueError(f"A taxa de payload deve estar em [0, 1]; recebido: {taxa_payload}.")
    return int(total * taxa_payload)


def inserir_lsb_sequencial(imagem: np.ndarray, taxa_payload: float, seed: int = 42) -> np.ndarray:
    """Grava bits aleatórios nos LSBs dos primeiros pixels, na ordem de varredura."""
    _validar_imagem(imagem)
    # copy() preserva a imagem original. reshape(-1) cria uma "visão" achatada: alterar "pixels"
    # altera "stego", sem copiar os dados de novo.
    stego = imagem.copy()
    pixels = stego.reshape(-1)
    n_bits = _bits_por_taxa(pixels.size, taxa_payload)
    bits = np.random.default_rng(seed).integers(0, 2, size=n_bits, dtype=np.uint8)
    pixels[:n_bits] = (pixels[:n_bits] & 0xFE) | bits
    return stego


def inserir_lsb_pseudoaleatorio(imagem: np.ndarray, taxa_payload: float, seed: int = 42) -> np.ndarray:
    """Grava bits aleatórios nos LSBs de pixels sorteados sem repetição pela semente."""
    _validar_imagem(imagem)
    stego = imagem.copy()
    pixels = stego.reshape(-1)
    n_bits = _bits_por_taxa(pixels.size, taxa_payload)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    # replace=False impede sortear o mesmo pixel duas vezes, o que apagaria um bit já gravado.
    posicoes = rng.choice(pixels.size, size=n_bits, replace=False)
    pixels[posicoes] = (pixels[posicoes] & 0xFE) | bits
    return stego


def _ordem_dos_pixels(total: int, metodo: str, chave: int) -> np.ndarray:
    """Ordem em que os bits são gravados: a chave define a permutação no modo pseudoaleatório."""
    if metodo == "sequencial":
        return np.arange(total)
    if metodo == "pseudoaleatorio":
        # A mesma chave sempre gera a mesma permutação. É isso que permite ao destinatário
        # visitar os pixels na mesma ordem em que eles foram escritos.
        return np.random.default_rng(chave).permutation(total)
    raise ValueError(f"Método desconhecido: {metodo}. Use um de {METODOS}.")


def inserir_mensagem(imagem: np.ndarray, mensagem: bytes, metodo: str = "sequencial", chave: int = 0) -> np.ndarray:
    """Esconde os bytes da mensagem, precedidos do tamanho, nos LSBs da imagem."""
    _validar_imagem(imagem)
    stego = imagem.copy()
    pixels = stego.reshape(-1)
    # Formato gravado: [4 bytes com o tamanho][bytes da mensagem].
    dados = len(mensagem).to_bytes(_BYTES_CABECALHO, "big") + mensagem
    # unpackbits transforma cada byte em 8 valores 0 ou 1, um para cada pixel.
    bits = np.unpackbits(np.frombuffer(dados, dtype=np.uint8))
    if bits.size > pixels.size:
        raise ValueError(
            f"Mensagem grande demais: precisa de {bits.size} bits e a imagem comporta {pixels.size}."
        )
    posicoes = _ordem_dos_pixels(pixels.size, metodo, chave)[: bits.size]
    pixels[posicoes] = (pixels[posicoes] & 0xFE) | bits
    return stego


def extrair_mensagem(stego: np.ndarray, metodo: str = "sequencial", chave: int = 0) -> bytes:
    """Recupera a mensagem gravada por ``inserir_mensagem`` com o mesmo método e chave."""
    _validar_imagem(stego)
    pixels = stego.reshape(-1)
    ordem = _ordem_dos_pixels(pixels.size, metodo, chave)
    bits_cabecalho = _BYTES_CABECALHO * 8
    if pixels.size < bits_cabecalho:
        raise ValueError("A imagem é pequena demais para conter o cabeçalho da mensagem.")
    # "& 1" lê só o último bit de cada pixel; packbits junta cada 8 bits de volta em um byte.
    # Primeiro lemos o cabeçalho, para descobrir quantos bytes vêm depois.
    tamanho = int.from_bytes(np.packbits(pixels[ordem[:bits_cabecalho]] & 1).tobytes(), "big")
    fim = bits_cabecalho + tamanho * 8
    # Com a chave ou o método errado, o "tamanho" lido é lixo e quase sempre não cabe na imagem.
    if fim > pixels.size:
        raise ValueError("Cabeçalho inválido: a imagem não contém mensagem com este método e chave.")
    return np.packbits(pixels[ordem[bits_cabecalho:fim]] & 1).tobytes()
