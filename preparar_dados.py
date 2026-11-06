"""Gera tudo o que existe em imagens/ e dados/ (exceto a mensagem de exemplo).

- imagens/kodak/: Kodak Lossless True Color Image Suite. Se faltar alguma imagem, ela é
  baixada e sua integridade é conferida por SHA-256.
- imagens/exemplos/: uma imagem limpa e seis esteganografadas, sempre com os mesmos bits.

Uso: python preparar_dados.py
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from entrada import carregar_cinza
from experimento import CENARIOS_PROJETO, semente

RAIZ = Path(__file__).resolve().parent
PASTA_KODAK = RAIZ / "imagens" / "kodak"
PASTA_EXEMPLOS = RAIZ / "imagens" / "exemplos"
# Endereço das imagens originais. {} recebe o nome do arquivo.
URL_KODAK = "https://r0k.us/graphics/kodak/kodak/{}"
# Imagem base dos exemplos. O restante do experimento usa as 24 imagens.
COVER_EXEMPLO = "kodim23.png"
# Faixa de sementes reservada aos exemplos, para não repetir as sementes do experimento.
_INDICE_EXEMPLOS = 700


def nome_exemplo(cenario) -> str:
    """Nome do arquivo de exemplo de um cenário, como sequencial_50.png."""
    if cenario.metodo is None:
        return "limpa.png"
    return f"{cenario.metodo}_{cenario.taxa * 100:g}.png"


def imagem_exemplo(cover: np.ndarray, indice: int, cenario) -> np.ndarray:
    """Gera a imagem de um cenário. A semente fixa faz o resultado ser sempre o mesmo."""
    return cenario.aplicar(cover, semente(_INDICE_EXEMPLOS, indice))


def _sha256(caminho: Path) -> str:
    """Calcula a impressão digital do arquivo: qualquer byte alterado muda o resultado."""
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def preparar_kodak() -> None:
    """Baixa as imagens que faltam, conferindo todas com a lista SHA256SUMS."""
    PASTA_KODAK.mkdir(parents=True, exist_ok=True)
    for linha in (PASTA_KODAK / "SHA256SUMS").read_text().splitlines():
        esperado, nome = linha.split()
        destino = PASTA_KODAK / nome
        if not destino.exists():
            print(f"Baixando {nome}...")
            with urllib.request.urlopen(URL_KODAK.format(nome), timeout=60) as resposta:
                destino.write_bytes(resposta.read())
        # O SHA-256 confirma que o arquivo é exatamente o mesmo usado nos resultados do relatório.
        if _sha256(destino) != esperado:
            destino.unlink()
            raise SystemExit(f"{nome}: SHA-256 não confere; arquivo removido.")
    print(f"Kodak conferida em {PASTA_KODAK}")


def gerar_exemplos() -> None:
    """Grava uma imagem limpa e as seis esteganografadas usadas na demonstração."""
    PASTA_EXEMPLOS.mkdir(parents=True, exist_ok=True)
    # Converte para cinza ANTES da inserção. Converter depois apagaria a mensagem.
    cover = carregar_cinza(PASTA_KODAK / COVER_EXEMPLO)
    for indice, cenario in enumerate(CENARIOS_PROJETO):
        destino = PASTA_EXEMPLOS / nome_exemplo(cenario)
        Image.fromarray(imagem_exemplo(cover, indice, cenario)).save(destino)
        print(f"Gerado {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    preparar_kodak()
    gerar_exemplos()
