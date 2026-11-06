"""Ataque do Qui-Quadrado sobre Pares de Valores (Westfeld & Pfitzmann), implementado do zero.

IDEIA DO ATAQUE
    Trocar o LSB de um pixel só alterna entre dois valores vizinhos: 2k <-> 2k+1. Esses dois
    valores formam um Par de Valores (PoV). Quando bits uniformes são gravados nos LSBs, cada
    pixel escolhido cai em um dos dois valores do par com a mesma chance, então as frequências
    do par se igualam. Em uma imagem limpa, em geral elas são diferentes.

HIPÓTESE NULA (H0)
    Dentro de cada par {2k, 2k+1}, os dois valores têm a mesma frequência, como acontece depois
    de uma inserção LSB completa. Atenção à inversão: aqui, H0 representa a imagem COM mensagem.
    Rejeitar H0 (p perto de 0) é evidência de que a imagem está limpa.

AS DUAS FORMAS DA ESTATÍSTICA
    - ``classica``: chi2 = soma de (h_2k - Z_k)^2 / Z_k, com K-1 graus de liberdade, onde
      Z_k = (h_2k + h_2k+1) / 2. É a fórmula do artigo original. Ela soma só uma das duas
      células de cada par, então seu valor esperado sob H0 é K/2, e não K-1. O resultado é
      um p próximo de 1 em imagens com inserção completa. Serve como indicador, mas não é
      um p-valor calibrado.
    - ``pearson``: soma as duas células (o dobro da clássica), com K graus de liberdade.
      É o teste de aderência correto: sob H0, p é aproximadamente uniforme em [0, 1].

    Em nenhuma das formas p é a probabilidade de a imagem conter mensagem.

EXEMPLO NUMÉRICO
    Par (10, 11) com contagens 30 e 10: Z = 20 e o termo vale (30 - 20)^2 / 20 = 5,0.
    Par (20, 21) com contagens 8 e 12:  Z = 10 e o termo vale (8 - 10)^2 / 10 = 0,4.
    chi2 clássico = 5,4 com K - 1 = 1 grau de liberdade. Pares desiguais geram chi2 alto e p baixo.

ORDEM DE LEITURA
    calcular_povs -> _chi2_dos_pares -> teste_chi_quadrado -> varredura_cumulativa -> decidir
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

FORMULAS = ("classica", "pearson")

# Um par só entra no teste se sua frequência esperada Z_k for ao menos 5. Abaixo disso, a
# distribuição chi-quadrado deixa de ser uma boa aproximação para a estatística.
MIN_ESPERADO_PADRAO = 5.0

# Número de pontos da varredura. Com 100, testamos 1%, 2%, ..., 100% dos pixels.
PASSOS_PADRAO = 100

# Regra de decisão usada por "--analisar". Estes valores não foram escolhidos à mão: vêm de
# experimento.calibrar_regra; ver resultados/calibracao.csv e RELATORIO.md.
LIMIAR_PADRAO = 0.999
PCT_MINIMO_PADRAO = 10.0


def _validar_amostras(amostras: np.ndarray) -> None:
    """Recusa qualquer entrada que não seja uint8, em vez de convertê-la sem avisar.

    Uma conversão automática com astype(uint8) transformaria 256 em 0 e -1 em 255, gerando um
    histograma errado sem nenhum sinal de erro.
    """
    if not isinstance(amostras, np.ndarray) or amostras.dtype != np.uint8:
        tipo = getattr(amostras, "dtype", type(amostras))
        raise TypeError(f"As amostras devem ser um ndarray uint8; recebido: {tipo}.")


def calcular_povs(amostras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna as frequências dos valores pares (2k) e ímpares (2k+1) do histograma."""
    _validar_amostras(amostras)
    # bincount conta quantas vezes cada valor aparece; minlength garante sempre 256 posições.
    hist = np.bincount(amostras.reshape(-1), minlength=256).astype(np.float64)
    # hist[0::2] pega as posições 0, 2, 4...; hist[1::2] pega 1, 3, 5...
    # Assim, pares[k] e impares[k] são exatamente as duas frequências do par {2k, 2k+1}.
    return hist[0::2], hist[1::2]


def _chi2_dos_pares(
    pares: np.ndarray, impares: np.ndarray, min_esperado: float, formula: str
) -> tuple[float, float]:
    """Calcula (estatística, p). Retorna (nan, nan) quando há menos de 2 pares utilizáveis."""
    if formula not in FORMULAS:
        raise ValueError(f"Fórmula desconhecida: {formula}. Use uma de {FORMULAS}.")

    # Sob H0, cada valor do par deveria ter metade do total do par: Z_k.
    esperado = (pares + impares) / 2.0

    # Máscara booleana que descarta os pares raros ou ausentes da imagem.
    ativos = esperado >= min_esperado
    k = int(np.sum(ativos))

    # Com menos de 2 pares, não há graus de liberdade. Retornar nan deixa claro que o teste não
    # pôde ser feito. Retornar p = 0 seria lido, erradamente, como "imagem limpa".
    if k < 2:
        return float("nan"), float("nan")

    # O coração do ataque: quanto cada par se afasta da igualdade, somado em todos os pares.
    classica = float(np.sum((pares[ativos] - esperado[ativos]) ** 2 / esperado[ativos]))

    # chi2.sf é a "função de sobrevivência", igual a 1 - CDF: a chance de um valor pelo menos
    # tão grande quanto o observado. É mais precisa que calcular 1 - cdf com valores extremos.
    if formula == "classica":
        return classica, float(chi2.sf(classica, k - 1))

    # Pearson soma as duas células de cada par. Por simetria, elas têm o mesmo valor, então a
    # soma é o dobro da clássica.
    return 2.0 * classica, float(chi2.sf(2.0 * classica, k))


def teste_chi_quadrado(
    amostras: np.ndarray,
    min_esperado: float = MIN_ESPERADO_PADRAO,
    formula: str = "classica",
) -> tuple[float, float]:
    """Aplica o teste às amostras. Retorna (estatística, p), ou (nan, nan) se não for possível."""
    pares, impares = calcular_povs(amostras)
    return _chi2_dos_pares(pares, impares, min_esperado, formula)


def varredura_cumulativa(
    imagem: np.ndarray,
    passos: int = PASSOS_PADRAO,
    min_esperado: float = MIN_ESPERADO_PADRAO,
    formula: str = "classica",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Testa os primeiros 1/passos, 2/passos, ..., 100% dos pixels, na ordem de varredura.

    Por que varrer: uma mensagem sequencial ocupa só o começo da imagem. O início fica com os
    pares equalizados (p perto de 1), e, quando a mensagem acaba, p despenca. Olhando apenas a
    imagem inteira, essa mensagem passaria despercebida.

    O histograma é acumulado a cada passo, então o custo não cresce com o número de passos
    além do próprio cálculo do teste. Retorna (percentuais, estatísticas, valores de p).
    """
    _validar_amostras(imagem)
    if passos < 1:
        raise ValueError(f"O número de passos deve ser positivo; recebido: {passos}.")

    # reshape(-1) "achata" a imagem em uma fila, linha por linha. Em RGB, a ordem é R, G, B, R...
    pixels = imagem.reshape(-1)

    # Arredonda para cima: o ponto de x% nunca tem menos que x% dos pixels.
    # Somar (passos - 1) antes da divisão inteira é a forma clássica de calcular o teto.
    limites = (np.arange(1, passos + 1) * pixels.size + passos - 1) // passos

    hist = np.zeros(256, dtype=np.float64)
    estatisticas = np.full(passos, np.nan)
    valores_p = np.full(passos, np.nan)

    inicio = 0
    for i, fim in enumerate(limites):
        # Soma ao histograma só os pixels novos deste trecho, sem recontar os anteriores.
        hist += np.bincount(pixels[inicio:fim], minlength=256)
        inicio = fim
        estatisticas[i], valores_p[i] = _chi2_dos_pares(hist[0::2], hist[1::2], min_esperado, formula)

    return limites / pixels.size * 100.0, estatisticas, valores_p


@dataclass(frozen=True)
class Decisao:
    """Resultado da regra de decisão aplicada a uma varredura."""

    rotulo: str  # "suspeita", "limpa" ou "inconclusiva"
    p_maximo: float  # maior p entre os pontos com amostra suficiente
    p_imagem_inteira: float  # p no ponto de 100%
    ultimo_percentual_suspeito: float  # estima onde a mensagem termina; nan se nenhum ponto atingiu o limiar


def decidir(
    percentuais: np.ndarray,
    valores_p: np.ndarray,
    limiar: float = LIMIAR_PADRAO,
    pct_minimo: float = PCT_MINIMO_PADRAO,
) -> Decisao:
    """Marca como suspeita a imagem em que algum ponto com amostra suficiente atinge o limiar.

    Procurar em toda a varredura, e não só em 100%, é o que permite encontrar uma mensagem
    sequencial que deixa a parte final da imagem intacta. Os pontos abaixo de ``pct_minimo``
    são ignorados: com poucos pixels, o teste não tem poder para rejeitar H0, e não rejeitar
    H0 é justamente o que se lê como "suspeita". Pontos sem teste possível (nan) também são
    ignorados; se não sobrar nenhum, o resultado é "inconclusiva", nunca "limpa".
    """
    # Pontos que podem votar: amostra grande o bastante e teste realizado.
    elegiveis = (percentuais >= pct_minimo) & ~np.isnan(valores_p)
    if not elegiveis.any():
        return Decisao("inconclusiva", float("nan"), float(valores_p[-1]), float("nan"))

    # nan_to_num troca nan por -1, valor que nunca atinge o limiar. O "&" com elegiveis descarta
    # os pontos pequenos demais.
    acima = elegiveis & (np.nan_to_num(valores_p, nan=-1.0) >= limiar)
    return Decisao(
        rotulo="suspeita" if acima.any() else "limpa",
        p_maximo=float(np.max(valores_p[elegiveis])),
        p_imagem_inteira=float(valores_p[-1]),
        # percentuais[acima] mantém só os pontos suspeitos; [-1] é o último deles.
        ultimo_percentual_suspeito=float(percentuais[acima][-1]) if acima.any() else float("nan"),
    )
