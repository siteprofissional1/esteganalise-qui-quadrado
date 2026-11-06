"""Gráficos do experimento. Os cenários vêm de experimento.py; nenhum nome é repetido aqui."""

from pathlib import Path

import matplotlib

# O backend Agg gera arquivos de imagem sem precisar de janela gráfica.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experimento import CENARIOS_PSEUDO, CENARIOS_SEQUENCIAIS, LIMPA, Resultados

# Paleta dos gráficos. A identidade de cada série também é dada por marcador e legenda, e não só
# pela cor, para o gráfico continuar legível para daltônicos e em impressão.
_TINTA = "#0b0b0b"
_TINTA_SECUNDARIA = "#52514e"
_GRADE = "#e1e0d9"
_EIXO = "#c3c2b7"
_SUPERFICIE = "#fcfcfb"
_NEUTRO = "#52514e"  # imagem limpa: referência, não uma série de payload
_CORES_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: validadas entre si
_MARCADORES = ("s", "^", "D")


def _estilizar(ax: plt.Axes) -> None:
    """Deixa grade e eixos discretos, para o destaque ficar com os dados."""
    ax.set_facecolor(_SUPERFICIE)
    ax.grid(True, axis="y", color=_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(_EIXO)
    ax.tick_params(colors=_TINTA_SECUNDARIA, length=0)
    ax.xaxis.label.set_color(_TINTA_SECUNDARIA)
    ax.yaxis.label.set_color(_TINTA_SECUNDARIA)
    ax.title.set_color(_TINTA)


def _figura(*args, **kwargs):
    """Cria a figura com o fundo padrão."""
    fig, eixos = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(_SUPERFICIE)
    return fig, eixos


def _rotular_limiar(ax, limiar: float) -> None:
    """Desenha a linha do limiar de decisão, com seu valor escrito."""
    ax.axhline(limiar, color=_TINTA_SECUNDARIA, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"limiar p = {limiar:g}", xy=(99, limiar), xytext=(0, -14), textcoords="offset points",
                ha="right", color=_TINTA_SECUNDARIA, fontsize=9)


def _desenhar_serie(ax, percentuais, valores_p, cor, marcador, rotulo, deslocamento=0) -> None:
    """Desenha uma curva de p com a faixa interquartil das execuções."""
    # Resume várias execuções em uma curva (mediana) e uma faixa com os 50% centrais dos valores.
    mediana = np.nanmedian(valores_p, axis=0)
    q1, q3 = np.nanquantile(valores_p, [0.25, 0.75], axis=0)
    ax.fill_between(percentuais, q1, q3, color=cor, alpha=0.18, linewidth=0)
    ax.plot(
        percentuais, mediana, color=cor, linewidth=2, marker=marcador, markersize=8,
        markevery=(deslocamento, max(1, len(percentuais) // 10)), markeredgecolor=_SUPERFICIE, markeredgewidth=1, label=rotulo,
    )


def plotar_comparativo(resultados: Resultados, caminho: Path) -> None:
    """p ao longo da varredura para cada caso: mediana entre execuções e faixa interquartil."""
    fig, eixos = _figura(1, 2, figsize=(12, 4.8), sharey=True)
    grupos = (
        ("LSB sequencial", CENARIOS_SEQUENCIAIS),
        ("LSB pseudoaleatório", CENARIOS_PSEUDO),
    )
    pct = resultados.percentuais
    # Marcadores defasados: as séries coincidem em p = 0 e, sem isso, uma cobriria a outra.
    passo = max(1, len(pct) // 40)
    for ax, (titulo, cenarios) in zip(eixos, grupos):
        for j, (cenario, cor, marcador) in enumerate(zip(cenarios, _CORES_SERIES, _MARCADORES)):
            rotulo = f"{cenario.taxa * 100:g}% de payload"
            _desenhar_serie(ax, pct, resultados.varreduras[cenario], cor, marcador, rotulo, (j + 1) * passo)
        _desenhar_serie(ax, pct, resultados.varreduras[LIMPA], _NEUTRO, "o", LIMPA.nome, 0)
        _rotular_limiar(ax, resultados.limiar)
        ax.set_title(titulo, loc="left", fontweight="bold")
        ax.set_xlabel("Parte da imagem analisada (%)")
        ax.set_xlim(0, 100.5)
        ax.set_ylim(-0.03, 1.05)
        _estilizar(ax)
    eixos[0].set_ylabel("p da fórmula clássica do χ²")
    eixos[1].legend(loc="center right", frameon=False, labelcolor=_TINTA_SECUNDARIA)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)


def plotar_deteccao_vs_payload(curva: list[dict], caminho: Path) -> None:
    """Taxa de detecção do LSB pseudoaleatório em função do payload, nas duas regras."""
    fig, ax = _figura(figsize=(7.5, 4.4))
    taxas = [linha["taxa"] * 100 for linha in curva]
    series = (
        ("deteccao_varredura", "Algum ponto da varredura", _CORES_SERIES[0], "o"),
        ("deteccao_imagem_inteira", "Só a imagem inteira", _CORES_SERIES[1], "s"),
    )
    for chave, rotulo, cor, marcador in series:
        valores = [linha[chave] * 100 for linha in curva]
        ax.plot(taxas, valores, color=cor, linewidth=2, marker=marcador, markersize=8,
                markeredgecolor=_SUPERFICIE, markeredgewidth=1, label=rotulo)
    ax.set_title("Detecção do LSB pseudoaleatório só aparece perto de 100%", loc="left", fontweight="bold")
    ax.set_xlabel("Taxa de payload (% dos pixels)")
    ax.set_ylabel("Execuções marcadas como suspeitas (%)")
    ax.set_xlim(0, 102)
    ax.set_ylim(-3, 105)
    ax.legend(loc="upper left", frameon=False, labelcolor=_TINTA_SECUNDARIA)
    _estilizar(ax)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)


def plotar_falsos_positivos(classes: list[dict], limiar: float, pct_minimo: float, caminho: Path) -> None:
    """Falsos positivos por classe limpa, com e sem amostra mínima, com a contagem em cada barra."""
    fig, ax = _figura(figsize=(8.5, 5))
    # barh desenha de baixo para cima; inverter mantém a ordem da tabela de cima para baixo.
    invertidas = classes[::-1]
    posicoes = np.arange(len(invertidas))
    altura = 0.36
    series = (
        ("falsos_positivos_sem_amostra_minima", "Sem amostra mínima", _CORES_SERIES[1], altura / 2 + 0.01),
        ("falsos_positivos", f"Com pelo menos {pct_minimo:g}% da imagem", _CORES_SERIES[0], -altura / 2 - 0.01),
    )
    for chave, rotulo, cor, deslocamento in series:
        valores = [linha[chave] / linha["imagens"] * 100 for linha in invertidas]
        ax.barh(posicoes + deslocamento, valores, height=altura, color=cor, label=rotulo)
        for y, linha, valor in zip(posicoes + deslocamento, invertidas, valores):
            ax.annotate(f"{linha[chave]}/{linha['imagens']}", xy=(valor, y), xytext=(5, 0), textcoords="offset points",
                        va="center", color=_TINTA_SECUNDARIA, fontsize=9)
    ax.set_yticks(posicoes, [linha["classe"] for linha in invertidas])
    ax.set_xlim(0, 112)
    ax.set_xlabel(f"Imagens limpas marcadas como suspeitas (%), com p >= {limiar:g}")
    ax.set_title("O falso positivo depende do tipo de imagem e da amostra mínima", loc="left", fontweight="bold")
    _estilizar(ax)
    ax.grid(True, axis="x", color=_GRADE, linewidth=0.8)
    ax.grid(False, axis="y")
    ax.legend(loc="center right", frameon=False, labelcolor=_TINTA_SECUNDARIA)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)


def plotar_analise(percentuais: np.ndarray, por_plano: dict[str, np.ndarray], limiar: float, titulo: str, caminho: Path) -> None:
    """Gráfico do modo --analisar: uma curva por plano da imagem analisada."""
    fig, ax = _figura(figsize=(7.5, 4.2))
    cores = (*_CORES_SERIES, "#eda100")
    marcadores = (*_MARCADORES, "o")
    for (nome, valores_p), cor, marcador in zip(por_plano.items(), cores, marcadores):
        ax.plot(percentuais, valores_p, color=cor, linewidth=2, marker=marcador, markersize=8,
                markevery=(0, max(1, len(percentuais) // 10)), markeredgecolor=_SUPERFICIE,
                markeredgewidth=1, label=nome)
    _rotular_limiar(ax, limiar)
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.set_xlabel("Parte da imagem analisada (%)")
    ax.set_ylabel("p da fórmula clássica do χ²")
    ax.set_xlim(0, 100.5)
    ax.set_ylim(-0.03, 1.05)
    if len(por_plano) > 1:
        ax.legend(loc="center right", frameon=False, labelcolor=_TINTA_SECUNDARIA)
    _estilizar(ax)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)
