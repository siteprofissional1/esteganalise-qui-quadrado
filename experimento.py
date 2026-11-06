"""Experimento do Projeto 3: cenários, regra de decisão, calibração e métricas.

Toda a aleatoriedade deriva de SEMENTE_BASE, então repetir a execução gera os mesmos números.

COMO O EXPERIMENTO FUNCIONA (ver rodar_experimento, no fim do arquivo)
    1. Lê as imagens de cobertura e as separa em calibração e avaliação.
    2. Gera os 7 cenários do projeto (limpa, sequencial e pseudoaleatório a 10/50/100%).
    3. Varre também 8 classes de imagens limpas, para medir o falso positivo.
    4. Usa só a calibração para escolher o limiar e a amostra mínima.
    5. Mede detecção e falso positivo nas imagens que não participaram da escolha.
    6. Faz comparações que explicam os resultados: fórmulas, passos, tipo de mensagem e RGB.
    7. Mede o PSNR entre cada imagem com mensagem e sua cobertura.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest, chi2

from chi_square import (
    MIN_ESPERADO_PADRAO,
    calcular_povs,
    decidir,
    teste_chi_quadrado,
    varredura_cumulativa,
)
from entrada import carregar_cinza, carregar_rgb
from lsb import inserir_lsb_pseudoaleatorio, inserir_lsb_sequencial

# Fixar a semente base faz todo o experimento ser reproduzível: outra pessoa obtém os mesmos números.
SEMENTE_BASE = 2026
# Taxas de payload pedidas no enunciado do projeto.
TAXAS_PROJETO = (0.10, 0.50, 1.00)
# Curva de payload: mais pontos perto de 100%, porque é lá que a detecção muda.
TAXAS_CURVA = (0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.98, 1.00)
# Mensagens sequenciais curtas: mostram o efeito do número de passos e da amostra mínima.
TAXAS_MENSAGEM_CURTA = (0.005, 0.01, 0.02, 0.05, 0.10)
# Valores testados na calibração. Se o escolhido cair na ponta da grade, ela deve ser ampliada,
# pois o melhor valor pode estar fora dela. Por isso a grade vai além de 0,99 e o mínimo vai além de 10%.
GRADE_LIMIARES = np.concatenate([np.round(np.arange(0.01, 1.0, 0.01), 2), [0.995, 0.999, 0.9995, 0.9999]])
GRADE_PCT_MINIMO = (0.0, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0)

# Associa o nome do método à função que o executa, evitando uma sequência de if/else.
_INSERCAO: dict[str, Callable[..., np.ndarray]] = {
    "sequencial": inserir_lsb_sequencial,
    "pseudoaleatorio": inserir_lsb_pseudoaleatorio,
}
_NOME_METODO = {"sequencial": "sequencial", "pseudoaleatorio": "pseudoaleatório"}


# frozen=True torna o objeto imutável e permite usá-lo como chave de dicionário, como em varreduras[cenario].
@dataclass(frozen=True)
class Cenario:
    """Um caso do experimento. A imagem limpa tem metodo=None e taxa=0."""

    metodo: str | None = None
    taxa: float = 0.0

    @property
    def nome(self) -> str:
        """Nome único do cenário, usado em tabelas e gráficos."""
        if self.metodo is None:
            return "Limpa"
        return f"LSB {_NOME_METODO[self.metodo]} ({self.taxa * 100:g}%)"

    def aplicar(self, imagem: np.ndarray, seed: int) -> np.ndarray:
        """Devolve a imagem deste cenário; a imagem limpa volta sem alteração."""
        if self.metodo is None:
            return imagem
        return _INSERCAO[self.metodo](imagem, self.taxa, seed=seed)


# Os 7 casos pedidos no projeto. O nome de cada um é gerado por Cenario.nome, em um só lugar.
LIMPA = Cenario()
CENARIOS_SEQUENCIAIS = tuple(Cenario("sequencial", t) for t in TAXAS_PROJETO)
CENARIOS_PSEUDO = tuple(Cenario("pseudoaleatorio", t) for t in TAXAS_PROJETO)
CENARIOS_PROJETO = (LIMPA, *CENARIOS_SEQUENCIAIS, *CENARIOS_PSEUDO)


def semente(*indices: int) -> int:
    """Deriva uma semente estável e independente para cada combinação de índices."""
    # Sementes 1, 2, 3... usadas em vários lugares gerariam os mesmos bits em cenários diferentes.
    # SeedSequence mistura os índices e evita essa dependência oculta entre experimentos.
    return int(np.random.SeedSequence([SEMENTE_BASE, *indices]).generate_state(1)[0])


# ---------- Imagens ----------
def listar_covers(pasta: str | Path) -> list[Path]:
    """Lista os PNGs em ordem alfabética, para a divisão entre calibração e avaliação ser sempre igual."""
    caminhos = sorted(Path(pasta).glob("*.png"))
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem PNG em {pasta}. Rode: python preparar_dados.py")
    return caminhos


def gerar_imagem_sintetica(altura: int = 512, largura: int = 512, seed: int = SEMENTE_BASE) -> np.ndarray:
    """Imagem suave com textura filtrada, sem nenhum viés imposto entre pares e ímpares."""
    rng = np.random.default_rng(seed)
    # Duas matrizes com as coordenadas de cada pixel, usadas para desenhar ondas suaves.
    y, x = np.mgrid[0:altura, 0:largura].astype(np.float64)
    yn, xn = y / altura, x / largura
    cena = (
        100.0
        + 50.0 * np.sin(2.0 * np.pi * (1.5 * xn + 0.8 * yn))
        + 35.0 * np.cos(2.0 * np.pi * (3.0 * yn - 1.5 * xn))
    )
    # Ruído suavizado, para o histograma ter variação sem a estrutura de uma foto real.
    textura = gaussian_filter(rng.normal(0.0, 25.0, size=(altura, largura)), sigma=3.0)
    return np.clip(np.round(cena + textura), 0, 255).astype(np.uint8)


def _com_ruido(imagem: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Soma ruído gaussiano e volta ao intervalo de 0 a 255."""
    # Simula ruído de sensor. O ruído espalha o histograma e deixa os pares vizinhos mais parecidos,
    # o que dificulta o teste. É uma das classes usadas para medir falso positivo.
    ruido = np.random.default_rng(seed).normal(0.0, sigma, size=imagem.shape)
    return np.clip(np.round(imagem + ruido), 0, 255).astype(np.uint8)


def _jpeg_decodificado(imagem: np.ndarray, qualidade: int) -> np.ndarray:
    """Devolve a imagem depois de passar por JPEG, como a maioria das fotos da internet."""
    # Salva em JPEG na memória e lê de volta: a imagem passa pela compressão com perdas.
    buffer = io.BytesIO()
    Image.fromarray(imagem).save(buffer, format="JPEG", quality=qualidade)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"), dtype=np.uint8)


def _reduzida(imagem: np.ndarray) -> np.ndarray:
    """Reduz a imagem à metade em cada dimensão."""
    altura, largura = imagem.shape
    # Reduzir a imagem é muito comum na prática e altera a estatística entre pixels vizinhos.
    img = Image.fromarray(imagem).resize((largura // 2, altura // 2), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


# ---------- Varreduras ----------
def varrer_cenario(
    covers: list[np.ndarray], cenario: Cenario, indice_cenario: int, repeticoes: int, passos: int, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Varre o cenário em todas as imagens e sementes.

    Retorna (percentuais, valores de p por execução e passo, índice da imagem de cada execução).
    A imagem limpa não depende de semente, então roda uma única vez por imagem.
    """
    # Sem inserção, repetir com outra semente daria exatamente o mesmo resultado.
    n_rep = 1 if cenario.metodo is None else repeticoes
    linhas, donos, percentuais = [], [], None
    for i, cover in enumerate(covers):
        for r in range(n_rep):
            stego = cenario.aplicar(cover, semente(i, indice_cenario, r))
            percentuais, _, p = varredura_cumulativa(stego, passos=passos, **kwargs)
            linhas.append(p)
            # Guarda a imagem de origem de cada execução, para separar calibração e avaliação depois.
            donos.append(i)
    return percentuais, np.array(linhas), np.array(donos)


def _suspeitas_varredura(valores_p: np.ndarray, percentuais: np.ndarray, limiar: float, pct_minimo: float) -> np.ndarray:
    """Regra principal: algum ponto com pelo menos pct_minimo da imagem atinge o limiar."""
    # Seleciona as colunas (pontos da varredura) com amostra suficiente.
    elegiveis = percentuais >= pct_minimo
    # max(axis=1) pega o maior p de cada execução: basta um ponto atingir o limiar.
    # Trocar nan por -1 faz com que pontos sem teste nunca disparem o alarme.
    return np.nan_to_num(valores_p[:, elegiveis], nan=-1.0).max(axis=1) >= limiar


def _suspeitas_inteira(valores_p: np.ndarray, limiar: float) -> np.ndarray:
    """Regra de comparação: olha só o p da imagem inteira (100%)."""
    return np.nan_to_num(valores_p[:, -1], nan=-1.0) >= limiar


def intervalo_exato(sucessos: int, total: int) -> tuple[float, float]:
    """Intervalo de confiança de 95% de Clopper-Pearson para uma proporção."""
    # Por que usar um intervalo: 0 erros em 24 imagens não prova taxa zero. O intervalo informa até
    # onde a taxa real ainda é compatível com os dados (neste caso, até cerca de 14%).
    ic = binomtest(sucessos, total).proportion_ci(confidence_level=0.95, method="exact")
    return float(ic.low), float(ic.high)


# ---------- Classes de imagens limpas ----------
def classes_limpas(covers: list[np.ndarray], n_sinteticas: int = 24) -> dict[str, list[np.ndarray]]:
    """Imagens sem mensagem, de naturezas diferentes. Nenhuma recebe inserção."""
    classes = {
        "Naturais (Kodak)": covers,
        "Reduzidas a 50% (bicúbica)": [_reduzida(c) for c in covers],
        "JPEG q=90 decodificado": [_jpeg_decodificado(c, 90) for c in covers],
    }
    # Ruído crescente, do quase imperceptível ao forte.
    for j, sigma in enumerate((1, 2, 5, 10)):
        classes[f"Ruído gaussiano σ={sigma}"] = [_com_ruido(c, sigma, semente(900, j, i)) for i, c in enumerate(covers)]
    classes["Sintéticas suaves"] = [gerar_imagem_sintetica(seed=semente(901, i)) for i in range(n_sinteticas)]
    return classes


# ---------- Calibração ----------
def calibrar_regra(
    percentuais: np.ndarray,
    varreduras: dict[Cenario, np.ndarray],
    donos: dict[Cenario, np.ndarray],
    p_limpas: dict[str, np.ndarray],
    indices_calibracao: list[int],
):
    """Escolhe limiar e amostra mínima usando só as imagens de calibração, pelo J de Youden.

    J = detecção média nos cenários com mensagem - falso positivo médio entre as classes limpas.
    Todas as classes pesam igual, para o resultado não depender só de imagens fáceis. Havendo
    empate no J máximo, fica com a menor amostra mínima empatada, que detecta mensagens mais
    curtas, e com o centro da faixa de limiares empatados, que tem mais folga.
    Retorna (limiar, pct_minimo, tabela com todas as combinações).
    """
    # Busca exaustiva: testa todas as combinações de amostra mínima e limiar e registra o resultado.
    tabela = []
    for pct_minimo in GRADE_PCT_MINIMO:
        for limiar in GRADE_LIMIARES:
            # Falso positivo de cada classe limpa, somente nas imagens de calibração.
            fprs = [
                float(_suspeitas_varredura(p[indices_calibracao], percentuais, limiar, pct_minimo).mean())
                for p in p_limpas.values()
            ]
            # Taxa de detecção de cada cenário com mensagem, também só na calibração.
            deteccoes = []
            for cenario, valores_p in varreduras.items():
                if cenario.metodo is not None:
                    mascara = np.isin(donos[cenario], indices_calibracao)
                    deteccoes.append(float(_suspeitas_varredura(valores_p[mascara], percentuais, limiar, pct_minimo).mean()))
            fpr, deteccao = float(np.mean(fprs)), float(np.mean(deteccoes))
            tabela.append(
                {"pct_minimo": pct_minimo, "limiar": float(limiar), "fpr_medio": fpr, "deteccao_media": deteccao, "j": deteccao - fpr}
            )

    # A tolerância 1e-12 evita que diferenças de arredondamento desfaçam um empate verdadeiro.
    j_max = max(linha["j"] for linha in tabela)
    empatados = [linha for linha in tabela if linha["j"] >= j_max - 1e-12]
    pct_escolhido = min(linha["pct_minimo"] for linha in empatados)
    limiares = [linha["limiar"] for linha in empatados if linha["pct_minimo"] == pct_escolhido]
    return float(np.round(np.median(limiares), 4)), pct_escolhido, tabela


# ---------- Métricas ----------
def metricas_cenarios(
    varreduras: dict[Cenario, np.ndarray],
    donos: dict[Cenario, np.ndarray],
    percentuais: np.ndarray,
    limiar: float,
    pct_minimo: float,
    indices: list[int],
) -> list[dict]:
    """Calcula as taxas de decisão de cada cenário nas imagens indicadas."""
    linhas = []
    for cenario, valores_p in varreduras.items():
        # Mantém só as execuções das imagens pedidas. As taxas principais usam imagens fora da calibração;
        # avaliar nas mesmas imagens da calibração daria um resultado otimista demais.
        subconjunto = valores_p[np.isin(donos[cenario], indices)]
        por_varredura = _suspeitas_varredura(subconjunto, percentuais, limiar, pct_minimo)
        por_inteira = _suspeitas_inteira(subconjunto, limiar)

        # No sequencial, o último ponto suspeito estima onde a mensagem termina. Aqui medimos esse erro.
        erro_fim = float("nan")
        if cenario.metodo == "sequencial":
            erros = [
                abs(decidir(percentuais, p, limiar, pct_minimo).ultimo_percentual_suspeito - cenario.taxa * 100)
                for p, s in zip(subconjunto, por_varredura)
                if s
            ]
            erro_fim = float(np.mean(erros)) if erros else float("nan")

        linhas.append(
            {
                "cenario": cenario.nome,
                "execucoes": int(subconjunto.shape[0]),
                "imagens": len(indices),
                "suspeitas_varredura": int(por_varredura.sum()),
                "taxa_varredura": float(por_varredura.mean()),
                "suspeitas_imagem_inteira": int(por_inteira.sum()),
                "taxa_imagem_inteira": float(por_inteira.mean()),
                "p_mediano_100pct": float(np.nanmedian(subconjunto[:, -1])),
                "erro_medio_fim_pontos_pct": erro_fim,
            }
        )
    return linhas


def falsos_positivos_por_classe(
    p_limpas: dict[str, np.ndarray], percentuais: np.ndarray, limiar: float, pct_minimo: float, indices: list[int]
) -> list[dict]:
    """Falso positivo por classe nas imagens indicadas, com intervalo exato de 95%."""
    linhas = []
    for nome, p in p_limpas.items():
        subconjunto = p[indices]
        suspeitas = _suspeitas_varredura(subconjunto, percentuais, limiar, pct_minimo)
        # Calcula as duas regras de comparação para mostrar o efeito da amostra mínima.
        sem_minimo = _suspeitas_varredura(subconjunto, percentuais, limiar, 0.0)
        inteira = _suspeitas_inteira(subconjunto, limiar)
        baixo, alto = intervalo_exato(int(suspeitas.sum()), len(indices))
        linhas.append(
            {
                "classe": nome,
                "imagens": len(indices),
                "falsos_positivos": int(suspeitas.sum()),
                "fpr": float(suspeitas.mean()),
                "ic95_inf": baixo,
                "ic95_sup": alto,
                "falsos_positivos_sem_amostra_minima": int(sem_minimo.sum()),
                "falsos_positivos_so_imagem_inteira": int(inteira.sum()),
            }
        )
    return linhas


# ---------- Payload pseudoaleatório: curva e previsão teórica ----------
def taxa_critica(chi2_limpa: float, k: int, limiar: float) -> float:
    """Menor taxa α com p >= limiar segundo chi2(α) = (1-α)²·chi2_0 + [1-(1-α)²]·K/2."""
    # isf é a inversa de sf: o maior chi2 que ainda dá p >= limiar. A inserção precisa baixar chi2 até aqui.
    corte = float(chi2.isf(limiar, k - 1))
    if chi2_limpa <= corte:
        return 0.0
    # A inserção completa leva chi2 a cerca de K/2. Se o corte estiver abaixo disso, nunca será atingido.
    if corte <= k / 2:
        return float("nan")  # o limiar exige mais do que a inserção completa entrega
    # Basta igualar chi2(α) ao corte e isolar α:
    # (1-α)² = (corte - K/2) / (chi2_0 - K/2)
    return 1.0 - float(np.sqrt((corte - k / 2) / (chi2_limpa - k / 2)))


def curva_payload(covers: list[np.ndarray], limiar: float, pct_minimo: float, repeticoes: int, passos: int):
    """Detecção do LSB pseudoaleatório por taxa, comparando χ² medido e previsto.

    Retorna (linhas da curva, linhas por imagem com a taxa crítica).
    """
    # K e chi2_0 de cada imagem limpa, que são as entradas da fórmula teórica.
    k_e_chi2 = []
    for cover in covers:
        pares, impares = calcular_povs(cover)
        k_e_chi2.append((int(np.sum((pares + impares) / 2 >= MIN_ESPERADO_PADRAO)), teste_chi_quadrado(cover)[0]))

    curva = []
    for j, taxa in enumerate(TAXAS_CURVA):
        cenario = Cenario("pseudoaleatorio", taxa)
        pct, p, _ = varrer_cenario(covers, cenario, 100 + j, repeticoes, passos)

        erros = []
        for i, (cover, (k, chi2_0)) in enumerate(zip(covers, k_e_chi2)):
            # Previsão teórica. A diferença dentro de cada par encolhe por (1-α), e como chi2 depende do
            # quadrado da diferença, a parte original encolhe por (1-α)². O restante vem do sorteio dos bits
            # e tende a K/2. Exemplo: α = 0,5, chi2_0 = 12836 e K = 124 dão cerca de 3255, ainda muito acima
            # de K, por isso p continua em 0.
            restante = (1 - taxa) ** 2
            previsto = restante * chi2_0 + (1 - restante) * k / 2
            medido = teste_chi_quadrado(cenario.aplicar(cover, semente(i, 100 + j, 0)))[0]
            erros.append(abs(medido - previsto) / medido)

        curva.append(
            {
                "taxa": taxa,
                "execucoes": int(p.shape[0]),
                "deteccao_varredura": float(_suspeitas_varredura(p, pct, limiar, pct_minimo).mean()),
                "deteccao_imagem_inteira": float(_suspeitas_inteira(p, limiar).mean()),
                "p_mediano_100pct": float(np.nanmedian(p[:, -1])),
                "erro_relativo_mediano_chi2": float(np.median(erros)),
            }
        )

    por_imagem = [
        {"imagem": i, "k": k, "chi2_limpa": chi2_0, "taxa_critica": taxa_critica(chi2_0, k, limiar)}
        for i, (k, chi2_0) in enumerate(k_e_chi2)
    ]
    return curva, por_imagem


# ---------- Imperceptibilidade ----------
def psnr(original: np.ndarray, alterada: np.ndarray) -> float:
    """PSNR em dB entre duas imagens de 8 bits: quanto maior, menor a diferença entre elas."""
    # float64 evita o erro clássico: em uint8, 3 - 5 daria 254 em vez de -2.
    mse = float(np.mean((original.astype(np.float64) - alterada.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse))


def medir_psnr(covers: list[np.ndarray], repeticoes: int) -> list[dict]:
    """Mede a imperceptibilidade de cada cenário com mensagem, comparando stego e cover.

    Previsão: com taxa α, metade dos pixels usados muda em 1 nível, então MSE = α/2 e
    PSNR = 10·log10(255² · 2/α), o que dá cerca de 51 dB com α = 1.
    """
    linhas = []
    for j, cenario in enumerate(c for c in CENARIOS_PROJETO if c.metodo is not None):
        valores = [
            psnr(cover, cenario.aplicar(cover, semente(800, i, j, r)))
            for i, cover in enumerate(covers)
            for r in range(repeticoes)
        ]
        linhas.append(
            {
                "cenario": cenario.nome,
                "execucoes": len(valores),
                "psnr_medio_db": float(np.mean(valores)),
                "psnr_minimo_db": float(np.min(valores)),
                "psnr_previsto_db": 10.0 * float(np.log10(255.0**2 * 2.0 / cenario.taxa)),
            }
        )
    return linhas


# ---------- Comparações metodológicas ----------
def comparar_formulas(covers: list[np.ndarray], repeticoes: int) -> list[dict]:
    """Mostra o que cada fórmula entrega na imagem inteira, limpa e com inserção completa."""
    linhas = []
    cenarios = (LIMPA, Cenario("pseudoaleatorio", 1.0))
    for formula in ("classica", "pearson"):
        for c_idx, cenario in enumerate(cenarios):
            razoes, valores_p = [], []
            n_rep = 1 if cenario.metodo is None else repeticoes
            for i, cover in enumerate(covers):
                for r in range(n_rep):
                    stego = cenario.aplicar(cover, semente(i, 200 + c_idx, r))
                    estatistica, p = teste_chi_quadrado(stego, formula=formula)
                    pares, impares = calcular_povs(stego)
                    k = int(np.sum((pares + impares) / 2 >= MIN_ESPERADO_PADRAO))
                    # Sob H0, esta razão deve dar cerca de 0,5 na fórmula clássica e cerca de 1 na de Pearson.
                    razoes.append(estatistica / k)
                    valores_p.append(p)
            valores_p = np.array(valores_p)
            linhas.append(
                {
                    "formula": formula,
                    "cenario": cenario.nome,
                    "execucoes": len(valores_p),
                    "chi2_sobre_k_mediana": float(np.median(razoes)),
                    "p_q05": float(np.quantile(valores_p, 0.05)),
                    "p_mediana": float(np.median(valores_p)),
                    "p_q95": float(np.quantile(valores_p, 0.95)),
                    "fracao_p_abaixo_0_05": float(np.mean(valores_p < 0.05)),
                }
            )
    return linhas


def comparar_min_esperado(covers: list[np.ndarray], limiar: float, pct_minimo: float, repeticoes: int, passos: int) -> list[dict]:
    """Compara o critério de frequência esperada mínima: 1 (antigo) e 5 (usual)."""
    linhas = []
    # O mínimo 1 era o critério da primeira versão; 5 é o critério usual do teste.
    for minimo in (1.0, 5.0):
        for c_idx, cenario in enumerate((LIMPA, Cenario("sequencial", 0.10), Cenario("pseudoaleatorio", 1.0))):
            pct, p, _ = varrer_cenario(covers, cenario, 300 + c_idx, repeticoes, passos, min_esperado=minimo)
            linhas.append(
                {
                    "min_esperado": minimo,
                    "cenario": cenario.nome,
                    "execucoes": int(p.shape[0]),
                    "taxa_suspeitas": float(_suspeitas_varredura(p, pct, limiar, pct_minimo).mean()),
                    "p_minimo_primeiro_passo": float(np.nanmin(p[:, 0])),
                }
            )
    return linhas


def comparar_passos(covers: list[np.ndarray], limiar: float, pct_minimo: float, repeticoes: int) -> list[dict]:
    """Mede o efeito do número de passos e da amostra mínima em mensagens sequenciais curtas."""
    casos = [LIMPA, *(Cenario("sequencial", t) for t in TAXAS_MENSAGEM_CURTA)]
    linhas = []
    for j, cenario in enumerate(casos):
        linha = {"cenario": cenario.nome}
        # Com 10 passos, o primeiro ponto já tem 10% da imagem, então uma mensagem de 5% fica misturada
        # com pixels limpos. Mais passos encontram mensagens menores, mas com amostras cada vez menores.
        for passos in (10, 100, 1000):
            pct, p, _ = varrer_cenario(covers, cenario, 400 + j, repeticoes, passos)
            linha[f"sem_minimo_{passos}_passos"] = float(_suspeitas_varredura(p, pct, limiar, 0.0).mean())
            if passos == 100:
                linha["com_minimo_100_passos"] = float(_suspeitas_varredura(p, pct, limiar, pct_minimo).mean())
        linhas.append(linha)
    return linhas


def comparar_tipo_de_mensagem(
    covers: list[np.ndarray], texto: bytes, limiar: float, pct_minimo: float, passos: int
) -> list[dict]:
    """Compara bits uniformes (mensagem cifrada) com texto puro na mesma fração da imagem.

    O teste supõe bits uniformes. Texto em UTF-8 tem proporção desigual de zeros e uns, o que
    impede a igualdade perfeita dentro dos pares.
    """
    linhas = []
    # Texto não tem zeros e uns na mesma proporção: em ASCII, o primeiro bit de cada byte é sempre 0.
    bits_texto = np.unpackbits(np.frombuffer(texto, dtype=np.uint8))
    for taxa in (0.10, 0.50, 1.00):
        contagens = {"uniforme": 0, "texto": 0}
        medianas = {"uniforme": [], "texto": []}
        for i, cover in enumerate(covers):
            n_bits = int(cover.size * taxa)
            uniforme = inserir_lsb_sequencial(cover, taxa, seed=semente(600, i, int(taxa * 100)))
            # np.resize repete o texto até preencher o trecho; os LSBs ficam com a mesma proporção de bits 1.
            repetido = np.resize(bits_texto, n_bits)
            com_texto = cover.copy()
            pixels = com_texto.reshape(-1)
            pixels[:n_bits] = (pixels[:n_bits] & 0xFE) | repetido
            for tipo, imagem in (("uniforme", uniforme), ("texto", com_texto)):
                pct, _, p = varredura_cumulativa(imagem, passos=passos)
                contagens[tipo] += decidir(pct, p, limiar, pct_minimo).rotulo == "suspeita"
                medianas[tipo].append(float(np.nanmax(p[pct >= pct_minimo])))
        linhas.append(
            {
                "taxa": taxa,
                "imagens": len(covers),
                "fracao_de_uns_no_texto": float(bits_texto.mean()),
                "suspeitas_bits_uniformes": int(contagens["uniforme"]),
                "suspeitas_texto_puro": int(contagens["texto"]),
                "p_max_mediano_uniforme": float(np.median(medianas["uniforme"])),
                "p_max_mediano_texto": float(np.median(medianas["texto"])),
            }
        )
    return linhas


# ---------- RGB: efeito da conversão para cinza ----------
# Os 4 planos analisados: cada canal e a sequência intercalada RGBRGB...
def _planos_rgb(rgb: np.ndarray) -> list[np.ndarray]:
    """Devolve os planos analisados em uma imagem RGB."""
    return [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], rgb]


def comparar_rgb(caminhos: list[Path], limiar: float, pct_minimo: float, passos: int) -> list[dict]:
    """Compara a análise por canal com a conversão para cinza usada na versão anterior."""
    casos: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]] = [
        ("Limpa", lambda rgb, s: rgb),
        ("Pseudoaleatório 100%, intercalado", lambda rgb, s: inserir_lsb_pseudoaleatorio(rgb, 1.0, seed=s)),
        ("Sequencial 50%, intercalado", lambda rgb, s: inserir_lsb_sequencial(rgb, 0.5, seed=s)),
    ]

    # Caso difícil: mensagem em um só canal. A sequência intercalada dilui o sinal; o canal azul isolado não.
    def so_azul(rgb: np.ndarray, s: int) -> np.ndarray:
        stego = rgb.copy()
        stego[:, :, 2] = inserir_lsb_pseudoaleatorio(rgb[:, :, 2], 1.0, seed=s)
        return stego

    casos.append(("Pseudoaleatório 100%, só canal azul", so_azul))

    covers = [carregar_rgb(c) for c in caminhos]
    linhas = []
    for c_idx, (nome, inserir) in enumerate(casos):
        por_canal = por_cinza = 0
        for i, rgb in enumerate(covers):
            stego = inserir(rgb, semente(i, 500 + c_idx))
            # A conversão faz uma média ponderada de R, G e B e arredonda: os LSBs originais se perdem.
            # É o que a primeira versão fazia, e por isso ela não detectava mensagens em RGB.
            cinza = np.asarray(Image.fromarray(stego).convert("L"), dtype=np.uint8)
            pct, _, p_cinza = varredura_cumulativa(cinza, passos=passos)
            por_cinza += decidir(pct, p_cinza, limiar, pct_minimo).rotulo == "suspeita"
            # A imagem é suspeita se qualquer plano for suspeito.
            por_canal += any(
                decidir(*_pct_e_p(plano, passos), limiar, pct_minimo).rotulo == "suspeita"
                for plano in _planos_rgb(stego)
            )
        linhas.append(
            {
                "caso": nome,
                "imagens": len(covers),
                "suspeitas_por_canal": int(por_canal),
                "suspeitas_convertendo_para_cinza": int(por_cinza),
            }
        )
    return linhas


def _pct_e_p(plano: np.ndarray, passos: int) -> tuple[np.ndarray, np.ndarray]:
    """Atalho que devolve só os percentuais e os valores de p, na ordem que decidir() espera."""
    pct, _, p = varredura_cumulativa(plano, passos=passos)
    return pct, p


# ---------- Experimento completo ----------
@dataclass
class Resultados:
    """Tudo o que main.py precisa para salvar tabelas e desenhar os gráficos."""

    percentuais: np.ndarray
    varreduras: dict[Cenario, np.ndarray]
    limiar: float
    pct_minimo: float
    tabelas: dict[str, list[dict]]


def rodar_experimento(
    pasta_covers: str | Path, texto: bytes, repeticoes: int = 5, passos: int = 100, avisar=print
) -> Resultados:
    """Executa todas as medições do relatório sobre as imagens da pasta."""
    caminhos = listar_covers(pasta_covers)
    covers = [carregar_cinza(c) for c in caminhos]
    # Separa as imagens: metade escolhe a regra, a outra metade mede o desempenho.
    # É a mesma ideia de treino e teste em aprendizado de máquina.
    calibracao = list(range(0, len(covers), 2))
    avaliacao = list(range(1, len(covers), 2))
    avisar(f"{len(covers)} imagens: {len(calibracao)} para calibrar, {len(avaliacao)} para avaliar; {repeticoes} sementes.")

    avisar("Varrendo os cenários do projeto...")
    varreduras, donos, percentuais = {}, {}, None
    for j, cenario in enumerate(CENARIOS_PROJETO):
        percentuais, varreduras[cenario], donos[cenario] = varrer_cenario(covers, cenario, j, repeticoes, passos)

    avisar("Varrendo as classes de imagens limpas...")
    p_limpas = {
        nome: np.array([varredura_cumulativa(img, passos=passos)[2] for img in imagens])
        for nome, imagens in classes_limpas(covers).items()
    }

    # A regra é escolhida aqui e, a partir deste ponto, não muda.
    limiar, pct_minimo, tabela_calibracao = calibrar_regra(percentuais, varreduras, donos, p_limpas, calibracao)
    avisar(f"Regra calibrada: suspeita se p >= {limiar:g} em algum ponto com pelo menos {pct_minimo:g}% da imagem.")

    tabelas: dict[str, list[dict]] = {
        "calibracao": tabela_calibracao,
        "metricas_por_cenario": metricas_cenarios(varreduras, donos, percentuais, limiar, pct_minimo, avaliacao),
        "metricas_por_cenario_todas_imagens": metricas_cenarios(
            varreduras, donos, percentuais, limiar, pct_minimo, calibracao + avaliacao
        ),
        "falsos_positivos_por_classe": falsos_positivos_por_classe(p_limpas, percentuais, limiar, pct_minimo, avaliacao),
        "falsos_positivos_por_classe_todas_imagens": falsos_positivos_por_classe(
            p_limpas, percentuais, limiar, pct_minimo, calibracao + avaliacao
        ),
    }
    avisar("Medindo a curva de payload e a previsão teórica...")
    tabelas["curva_payload"], tabelas["taxa_critica_por_imagem"] = curva_payload(covers, limiar, pct_minimo, repeticoes, passos)
    avisar("Comparando fórmulas, frequência mínima, passos, tipo de mensagem e RGB...")
    tabelas["psnr"] = medir_psnr(covers, repeticoes)
    tabelas["comparacao_formulas"] = comparar_formulas(covers, repeticoes)
    tabelas["comparacao_min_esperado"] = comparar_min_esperado(covers, limiar, pct_minimo, repeticoes, passos)
    tabelas["comparacao_passos"] = comparar_passos(covers, limiar, pct_minimo, repeticoes)
    tabelas["comparacao_tipo_de_mensagem"] = comparar_tipo_de_mensagem(covers, texto, limiar, pct_minimo, passos)
    tabelas["comparacao_rgb"] = comparar_rgb(caminhos, limiar, pct_minimo, passos)

    return Resultados(percentuais, varreduras, limiar, pct_minimo, tabelas)
