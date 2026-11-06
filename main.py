"""Projeto 3: esteganálise pelo teste do Qui-Quadrado.

Uso:
  python main.py                       roda o experimento completo e salva tabelas e gráficos
  python main.py --analisar IMG        diz se IMG é suspeita, sem alterar nada nela
  python main.py --inserir COVER --saida STEGO --mensagem ARQ [--metodo M] [--chave N]
  python main.py --extrair STEGO [--metodo M] [--chave N]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from chi_square import LIMIAR_PADRAO, PASSOS_PADRAO, PCT_MINIMO_PADRAO, decidir, varredura_cumulativa
from entrada import carregar_para_analise
from experimento import CENARIOS_PROJETO, rodar_experimento
from graficos_exp import plotar_analise, plotar_comparativo, plotar_deteccao_vs_payload, plotar_falsos_positivos
from lsb import METODOS, extrair_mensagem, inserir_mensagem

# Caminhos relativos a este arquivo: o programa funciona a partir de qualquer pasta.
RAIZ = Path(__file__).resolve().parent
PASTA_KODAK = RAIZ / "imagens" / "kodak"
ARQUIVO_MENSAGEM = RAIZ / "dados" / "mensagem.txt"
COLUNAS_TABELA = 10


def imprimir_tabela(titulo: str, percentuais: np.ndarray, linhas: dict[str, np.ndarray]) -> None:
    """Imprime p ao longo da varredura em até 10 colunas, com o cabeçalho vindo dos dados."""
    n = len(percentuais)
    # Escolhe até 10 colunas igualmente espaçadas, para a tabela caber no terminal.
    indices = np.linspace(n / min(COLUNAS_TABELA, n) - 1, n - 1, min(COLUNAS_TABELA, n)).round().astype(int)
    largura = max(len(nome) for nome in linhas)
    cabecalho = f"{'':<{largura}} | " + " | ".join(f"{percentuais[i]:>5.0f}%" for i in indices)
    print(f"\n{titulo}\n{cabecalho}\n{'-' * len(cabecalho)}")
    for nome, valores in linhas.items():
        print(f"{nome:<{largura}} | " + " | ".join(f"{valores[i]:6.4f}" for i in indices))


def analisar(caminho: str, limiar: float, passos: int, grafico: str | None, pct_minimo: float = PCT_MINIMO_PADRAO) -> None:
    """Analisa a imagem como está em disco. Nenhum bit é inserido ou alterado."""
    imagem = carregar_para_analise(caminho)
    print(f"Imagem: {caminho} (formato {imagem.formato}, modo {imagem.modo})")
    for aviso in imagem.avisos:
        print(f"Aviso: {aviso}")

    # Cada plano (L, ou R, G, B e RGB intercalado) recebe sua própria varredura e decisão.
    percentuais, por_plano, decisoes = None, {}, {}
    for nome, plano in imagem.planos.items():
        percentuais, _, valores_p = varredura_cumulativa(plano, passos=passos)
        por_plano[nome] = valores_p
        decisoes[nome] = decidir(percentuais, valores_p, limiar, pct_minimo)

    regra = f"suspeita se p >= {limiar:g} com pelo menos {pct_minimo:g}% da imagem"
    imprimir_tabela(f"p ao longo da varredura ({regra})", percentuais, por_plano)

    print("\nPlano | decisão | p máximo | p na imagem inteira | último ponto suspeito")
    for nome, d in decisoes.items():
        ultimo = "-" if np.isnan(d.ultimo_percentual_suspeito) else f"{d.ultimo_percentual_suspeito:.0f}%"
        print(f"{nome} | {d.rotulo} | {d.p_maximo:.4f} | {d.p_imagem_inteira:.4f} | {ultimo}")

    # Basta um plano suspeito para marcar a imagem. "Limpa" exige que todos os planos sejam limpos.
    rotulos = {d.rotulo for d in decisoes.values()}
    if "suspeita" in rotulos:
        final = "SUSPEITA: distribuição compatível com inserção LSB"
    elif rotulos == {"limpa"}:
        final = "SEM EVIDÊNCIA de LSB detectável (não prova ausência: inserção espalhada ou curta escapa do teste)"
    else:
        final = "INCONCLUSIVA: amostras insuficientes para o teste"
    print(f"\nResultado: {final}")

    if grafico:
        plotar_analise(percentuais, por_plano, limiar, Path(caminho).name, Path(grafico))
        print(f"Gráfico salvo em: {grafico}")


def inserir(cover: str, saida: str, mensagem: str, metodo: str, chave: int) -> None:
    """Esconde o arquivo de mensagem na imagem de cobertura e salva o resultado."""
    # Compressão com perdas altera os pixels e destruiria a mensagem gravada nos LSBs.
    if Path(saida).suffix.lower() not in (".png", ".bmp", ".tif", ".tiff"):
        raise SystemExit("A saída deve ser um formato sem perdas (.png, .bmp ou .tif): JPEG destruiria os LSBs.")
    with Image.open(cover) as img:
        modo_saida = "L" if img.mode in ("L", "LA") else "RGB"
        arr = np.asarray(img.convert(modo_saida), dtype=np.uint8)
    dados = Path(mensagem).read_bytes()
    stego = inserir_mensagem(arr, dados, metodo, chave)
    Image.fromarray(stego).save(saida)
    print(f"{len(dados)} bytes inseridos em {saida} (método {metodo}; {(len(dados) + 4) * 8 / arr.size:.2%} da capacidade).")


def extrair(stego: str, metodo: str, chave: int, saida: str | None) -> None:
    """Recupera a mensagem e a salva em arquivo ou a mostra no terminal."""
    with Image.open(stego) as img:
        arr = np.asarray(img, dtype=np.uint8)
    dados = extrair_mensagem(arr, metodo, chave)
    if saida:
        Path(saida).write_bytes(dados)
        print(f"{len(dados)} bytes extraídos para {saida}.")
    else:
        sys.stdout.write(dados.decode("utf-8", errors="replace") + "\n")


def _salvar_csv(linhas: list[dict], caminho: Path) -> None:
    """Salva uma lista de dicionários como CSV; as chaves viram o cabeçalho."""
    def formatar(v):
        # Limita a 6 algarismos significativos, para os CSVs ficarem legíveis e estáveis.
        return f"{v:.6g}" if isinstance(v, float) else v

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows({k: formatar(v) for k, v in linha.items()} for linha in linhas)


def rodar(pasta_covers: Path, repeticoes: int, passos: int) -> None:
    """Roda o experimento, salva tabelas e gráficos e imprime um resumo."""
    pasta_graficos = RAIZ / "graficos"
    pasta_resultados = RAIZ / "resultados"
    pasta_graficos.mkdir(exist_ok=True)
    pasta_resultados.mkdir(exist_ok=True)

    resultados = rodar_experimento(pasta_covers, ARQUIVO_MENSAGEM.read_bytes(), repeticoes, passos)

    for nome, linhas in resultados.tabelas.items():
        _salvar_csv(linhas, pasta_resultados / f"{nome}.csv")
    plotar_comparativo(resultados, pasta_graficos / "comparativo_chi_quadrado.png")
    plotar_deteccao_vs_payload(resultados.tabelas["curva_payload"], pasta_graficos / "deteccao_vs_payload.png")
    plotar_falsos_positivos(
        resultados.tabelas["falsos_positivos_por_classe"],
        resultados.limiar,
        resultados.pct_minimo,
        pasta_graficos / "falsos_positivos_por_classe.png",
    )

    # Mediana entre execuções, ponto a ponto. É menos sensível a valores extremos que a média.
    medianas = {c.nome: np.nanmedian(resultados.varreduras[c], axis=0) for c in CENARIOS_PROJETO}
    imprimir_tabela("p mediano ao longo da varredura", resultados.percentuais, medianas)

    print(
        f"\nDecisões nas imagens de avaliação (suspeita se p >= {resultados.limiar:g} "
        f"com pelo menos {resultados.pct_minimo:g}% da imagem)"
    )
    for linha in resultados.tabelas["metricas_por_cenario"]:
        print(
            f"{linha['cenario']:<32} varredura: {linha['suspeitas_varredura']:>3}/{linha['execucoes']:<3}"
            f"  imagem inteira: {linha['suspeitas_imagem_inteira']:>3}/{linha['execucoes']}"
        )

    print("\nFalsos positivos por classe de imagem limpa (imagens de avaliação)")
    for linha in resultados.tabelas["falsos_positivos_por_classe"]:
        print(
            f"{linha['classe']:<30} {linha['falsos_positivos']:>2}/{linha['imagens']}"
            f"  IC 95%: {linha['ic95_inf']:.0%} a {linha['ic95_sup']:.0%}"
        )

    # Avisa se a calibração passar a discordar dos valores fixados em chi_square.py.
    if (resultados.limiar, resultados.pct_minimo) != (LIMIAR_PADRAO, PCT_MINIMO_PADRAO):
        print(
            f"\nAtenção: a regra calibrada (p >= {resultados.limiar:g}, {resultados.pct_minimo:g}%) difere dos "
            f"valores padrão de --analisar (p >= {LIMIAR_PADRAO:g}, {PCT_MINIMO_PADRAO:g}%)."
        )
    print(f"\nTabelas em {pasta_resultados}\nGráficos em {pasta_graficos}")


def _argumentos() -> argparse.Namespace:
    """Define as opções da linha de comando."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Só um modo por vez. Sem nenhum deles, o programa roda o experimento completo.
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--analisar", metavar="IMG", help="analisa IMG como está, sem alterá-la")
    modo.add_argument("--inserir", metavar="COVER", help="esconde uma mensagem na imagem COVER")
    modo.add_argument("--extrair", metavar="STEGO", help="recupera a mensagem da imagem STEGO")
    p.add_argument("--limiar", type=float, default=LIMIAR_PADRAO, help="--analisar: suspeita se p >= limiar")
    p.add_argument("--pct-minimo", type=float, default=PCT_MINIMO_PADRAO, help="--analisar: ignora pontos com menos que este %% da imagem")
    p.add_argument("--passos", type=int, default=PASSOS_PADRAO, help="número de pontos da varredura")
    p.add_argument("--grafico", metavar="ARQ", help="--analisar: salva o gráfico da varredura")
    p.add_argument("--saida", metavar="ARQ", help="--inserir: imagem gerada; --extrair: arquivo da mensagem")
    p.add_argument("--mensagem", metavar="ARQ", help="--inserir: arquivo com a mensagem")
    p.add_argument("--metodo", choices=METODOS, default="sequencial")
    p.add_argument("--chave", type=int, default=0, help="chave do método pseudoaleatório")
    p.add_argument("--covers", type=Path, default=PASTA_KODAK, help="experimento: pasta das imagens de cobertura")
    p.add_argument("--repeticoes", type=int, default=5, help="experimento: sementes por imagem e cenário")
    return p.parse_args()


def main() -> None:
    """Encaminha para o modo pedido na linha de comando."""
    args = _argumentos()
    try:
        if args.analisar:
            analisar(args.analisar, args.limiar, args.passos, args.grafico, args.pct_minimo)
        elif args.inserir:
            if not (args.saida and args.mensagem):
                raise SystemExit("--inserir exige --saida e --mensagem.")
            inserir(args.inserir, args.saida, args.mensagem, args.metodo, args.chave)
        elif args.extrair:
            extrair(args.extrair, args.metodo, args.chave, args.saida)
        else:
            rodar(args.covers, args.repeticoes, args.passos)
    # Arquivo inexistente ou imagem inválida viram uma mensagem clara, em vez de um traceback.
    except (OSError, ValueError) as erro:
        raise SystemExit(f"Erro: {erro}") from erro


if __name__ == "__main__":
    main()
