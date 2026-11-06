"""Testes do detector, da inserção e da análise. Rode com: python -m pytest

Os testes também servem como exemplos de uso de cada função. Vários deles protegem contra um
erro específico da primeira versão do projeto, para que ele não volte sem ser percebido.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.stats import chi2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from chi_square import calcular_povs, decidir, teste_chi_quadrado, varredura_cumulativa  # noqa: E402
from entrada import carregar_para_analise  # noqa: E402
from experimento import CENARIOS_PROJETO, gerar_imagem_sintetica, psnr, taxa_critica  # noqa: E402
import main as cli  # noqa: E402
from lsb import extrair_mensagem, inserir_lsb_pseudoaleatorio, inserir_lsb_sequencial, inserir_mensagem  # noqa: E402
from preparar_dados import COVER_EXEMPLO, PASTA_EXEMPLOS, PASTA_KODAK, imagem_exemplo, nome_exemplo  # noqa: E402


def _imagem_desequilibrada(tamanho: int = 256) -> np.ndarray:
    """Só valores pares, o caso mais distante de H0 possível."""
    rng = np.random.default_rng(1)
    return (rng.integers(50, 100, size=(tamanho, tamanho), dtype=np.uint8) * 2).astype(np.uint8)


# ---------- Qui-quadrado ----------
# Confere a fórmula contra uma conta feita à mão: é o teste mais importante do projeto.
def test_calculo_manual_da_formula_classica():
    # Par (10, 11) com 30 e 10: Z=20, (30-20)²/20=5. Par (20, 21) com 8 e 12: Z=10, (8-10)²/10=0.4.
    amostras = np.array([10] * 30 + [11] * 10 + [20] * 8 + [21] * 12, dtype=np.uint8)
    estatistica, p = teste_chi_quadrado(amostras)
    assert estatistica == pytest.approx(5.4)
    assert p == pytest.approx(chi2.sf(5.4, 1))


def test_pearson_e_o_dobro_da_classica_com_k_graus():
    amostras = np.array([10] * 30 + [11] * 10 + [20] * 8 + [21] * 12, dtype=np.uint8)
    estatistica, p = teste_chi_quadrado(amostras, formula="pearson")
    assert estatistica == pytest.approx(10.8)
    assert p == pytest.approx(chi2.sf(10.8, 2))


def test_criterio_de_frequencia_esperada():
    # O par (20, 21) tem Z=3: fica de fora com mínimo 5, sobrando 2 pares ativos.
    amostras = np.array([10] * 30 + [11] * 10 + [20] * 3 + [21] * 3 + [40] * 6 + [41] * 6, dtype=np.uint8)
    assert teste_chi_quadrado(amostras, min_esperado=5)[1] == pytest.approx(chi2.sf(5.0, 1))
    assert teste_chi_quadrado(amostras, min_esperado=1)[1] == pytest.approx(chi2.sf(5.0, 2))


@pytest.mark.parametrize(
    "amostras",
    [np.full((64, 64), 7, dtype=np.uint8), np.array([], dtype=np.uint8), np.array([1, 2, 3], dtype=np.uint8)],
)
def test_sem_teste_possivel_retorna_nan(amostras):
    assert all(np.isnan(v) for v in teste_chi_quadrado(amostras))


@pytest.mark.parametrize("entrada", [np.array([256, 257, -1]), np.array([1.5, 2.5]), [1, 2, 3]])
def test_tipos_errados_sao_recusados(entrada):
    with pytest.raises(TypeError):
        teste_chi_quadrado(entrada)


def test_varredura_coincide_com_teste_direto_em_cada_passo():
    imagem = gerar_imagem_sintetica(128, 128)
    pct, estatisticas, valores_p = varredura_cumulativa(imagem, passos=7)
    assert len(pct) == 7 and pct[-1] == 100.0
    pixels = imagem.reshape(-1)
    for i, percentual in enumerate(pct):
        limite = round(percentual / 100 * pixels.size)
        estatistica, p = teste_chi_quadrado(pixels[:limite])
        assert estatisticas[i] == pytest.approx(estatistica, nan_ok=True)
        assert valores_p[i] == pytest.approx(p, nan_ok=True)


# Simula o caso em que H0 é verdadeira e confirma a diferença entre as fórmulas (ver RELATORIO.md).
def test_h0_verdadeira_separa_as_duas_formulas():
    """Com pares equalizados, a fórmula clássica fica perto de 1 e a de Pearson, uniforme."""
    rng = np.random.default_rng(0)
    classica, pearson = [], []
    for _ in range(300):
        cheios = rng.integers(0, 256, size=200_000, dtype=np.uint8)
        classica.append(teste_chi_quadrado(cheios)[1])
        pearson.append(teste_chi_quadrado(cheios, formula="pearson")[1])
    assert np.quantile(classica, 0.05) > 0.95
    assert 0.40 < np.mean(pearson) < 0.60
    assert 0.02 < np.mean(np.array(pearson) < 0.05) < 0.09


# ---------- Decisão ----------
def test_decisao_tem_tres_estados():
    pct = np.array([50.0, 100.0])
    assert decidir(pct, np.array([0.0, 0.0]), 0.5, 0.0).rotulo == "limpa"
    assert decidir(pct, np.array([0.9, 0.0]), 0.5, 0.0).rotulo == "suspeita"
    assert decidir(pct, np.array([np.nan, np.nan]), 0.5, 0.0).rotulo == "inconclusiva"


def test_amostra_minima_ignora_pontos_iniciais():
    pct = np.array([1.0, 2.0, 50.0, 100.0])
    p = np.array([0.99, 0.99, 0.0, 0.0])
    assert decidir(pct, p, 0.5, 0.0).rotulo == "suspeita"
    assert decidir(pct, p, 0.5, 5.0).rotulo == "limpa"


# Tamanho 101 x 97 de propósito: o total de pixels não é divisível por 100 e expõe erros de arredondamento.
def test_ponto_de_x_por_cento_nunca_tem_menos_que_x_por_cento():
    imagem = gerar_imagem_sintetica(101, 97)
    pct, _, _ = varredura_cumulativa(imagem, passos=100)
    assert np.all(pct >= np.arange(1, 101) - 1e-9)


def test_decisao_ignora_nan_e_informa_o_ultimo_ponto_suspeito():
    d = decidir(np.array([25.0, 50.0, 75.0, 100.0]), np.array([np.nan, 0.8, 0.7, 0.0]), 0.5, 0.0)
    assert d.rotulo == "suspeita"
    assert d.ultimo_percentual_suspeito == 50.0 + 25.0
    assert d.p_maximo == pytest.approx(0.8)


# ---------- Inserção ----------
def test_insercao_sequencial_altera_so_o_prefixo_e_so_o_lsb():
    imagem = gerar_imagem_sintetica(64, 64)
    stego = inserir_lsb_sequencial(imagem, 0.25, seed=3)
    diferenca = (stego.astype(int) - imagem.astype(int)).reshape(-1)
    corte = imagem.size // 4
    assert np.all(diferenca[corte:] == 0)
    assert np.all(np.abs(diferenca) <= 1)
    assert 0.4 < np.mean(diferenca[:corte] != 0) < 0.6


def test_insercao_pseudoaleatoria_altera_a_fracao_certa_e_preserva_o_original():
    imagem = gerar_imagem_sintetica(128, 128)
    copia = imagem.copy()
    stego = inserir_lsb_pseudoaleatorio(imagem, 0.5, seed=3)
    assert np.array_equal(imagem, copia)
    assert np.all(np.abs(stego.astype(int) - imagem.astype(int)) <= 1)
    assert 0.20 < np.mean(stego != imagem) < 0.30


@pytest.mark.parametrize("taxa", [-0.1, 1.1])
def test_taxa_invalida_e_recusada(taxa):
    with pytest.raises(ValueError):
        inserir_lsb_sequencial(gerar_imagem_sintetica(16, 16), taxa)


@pytest.mark.parametrize("funcao", [inserir_lsb_sequencial, inserir_lsb_pseudoaleatorio])
def test_insercao_recusa_imagem_que_nao_e_uint8(funcao):
    with pytest.raises(TypeError):
        funcao(np.zeros((8, 8), dtype=np.float64), 0.5)


@pytest.mark.parametrize("metodo", ["sequencial", "pseudoaleatorio"])
def test_mensagem_ida_e_volta(metodo):
    imagem = gerar_imagem_sintetica(64, 64)
    mensagem = "Fuga às 22h, portão leste.".encode("utf-8")
    stego = inserir_mensagem(imagem, mensagem, metodo, chave=7)
    assert extrair_mensagem(stego, metodo, chave=7) == mensagem


def test_mensagem_maior_que_a_capacidade_e_recusada():
    with pytest.raises(ValueError):
        inserir_mensagem(np.zeros((8, 8), dtype=np.uint8), b"x" * 100)


# ---------- Detecção de ponta a ponta ----------
def test_detector_separa_os_casos_didaticos():
    limpa = _imagem_desequilibrada()
    pct, _, p = varredura_cumulativa(limpa, passos=20)
    assert decidir(pct, p, 0.5, 0.0).rotulo == "limpa"

    pct, _, p = varredura_cumulativa(inserir_lsb_pseudoaleatorio(limpa, 1.0, seed=5), passos=20)
    assert decidir(pct, p, 0.5, 0.0).rotulo == "suspeita"

    pct, _, p = varredura_cumulativa(inserir_lsb_sequencial(limpa, 0.5, seed=5), passos=20)
    d = decidir(pct, p, 0.5, 0.0)
    assert d.rotulo == "suspeita"
    assert 40 <= d.ultimo_percentual_suspeito <= 60


def test_taxa_critica_recupera_o_ponto_de_p_igual_ao_limiar():
    k, chi2_0, limiar = 100, 20_000.0, 0.5
    alfa = taxa_critica(chi2_0, k, limiar)
    restante = (1 - alfa) ** 2
    previsto = restante * chi2_0 + (1 - restante) * k / 2
    assert chi2.sf(previsto, k - 1) == pytest.approx(limiar)


# ---------- PSNR ----------
def test_psnr_calculado_a_mao():
    # Todos os pixels diferem em 1 nível: MSE = 1 e PSNR = 10·log10(255²) = 48,13 dB.
    a = np.full((8, 8), 100, dtype=np.uint8)
    assert psnr(a, a + 1) == pytest.approx(48.1308, abs=1e-3)
    # A diferença não pode dar a volta no uint8: 100 - 101 é -1, e não 255.
    assert psnr(a + 1, a) == pytest.approx(48.1308, abs=1e-3)


def test_psnr_de_imagens_iguais_e_infinito():
    a = gerar_imagem_sintetica(16, 16)
    assert psnr(a, a) == float("inf")


def test_psnr_da_insercao_completa_coincide_com_a_previsao():
    imagem = gerar_imagem_sintetica(256, 256)
    assert psnr(imagem, inserir_lsb_pseudoaleatorio(imagem, 1.0, seed=4)) == pytest.approx(51.14, abs=0.05)


# ---------- Leitura e CLI ----------
def test_rgb_e_lido_por_canal_sem_conversao(tmp_path):
    rgb = np.dstack([gerar_imagem_sintetica(64, 64, seed=s) for s in (1, 2, 3)])
    caminho = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(caminho)
    carregada = carregar_para_analise(str(caminho))
    assert list(carregada.planos) == ["R", "G", "B", "RGB intercalado"]
    assert np.array_equal(carregada.planos["B"], rgb[:, :, 2])
    assert np.array_equal(carregada.planos["RGB intercalado"], rgb)


# Garante que o erro da primeira versão (convert("L") apagando os LSBs) não volte.
def test_analise_de_lsb_em_rgb_sobrevive_a_leitura(tmp_path):
    base = np.dstack([_imagem_desequilibrada(128)] * 3)
    stego = inserir_lsb_pseudoaleatorio(base, 1.0, seed=2)
    caminho = tmp_path / "stego.png"
    Image.fromarray(stego).save(caminho)
    plano = carregar_para_analise(str(caminho)).planos["RGB intercalado"]
    pct, _, p = varredura_cumulativa(plano, passos=10)
    assert decidir(pct, p, 0.5, 0.0).rotulo == "suspeita"


def test_modo_sem_8_bits_e_recusado(tmp_path):
    caminho = tmp_path / "bin.png"
    Image.new("1", (8, 8)).save(caminho)
    with pytest.raises(ValueError):
        carregar_para_analise(str(caminho))


# Garante que a análise é somente leitura. A primeira versão inseria bits na imagem recebida.
def test_analisar_nao_altera_o_arquivo(tmp_path, capsys):
    caminho = tmp_path / "limpa.png"
    Image.fromarray(_imagem_desequilibrada(64)).save(caminho)
    antes = caminho.read_bytes()
    cli.analisar(str(caminho), 0.5, 10, None)
    assert caminho.read_bytes() == antes
    assert "SEM EVIDÊNCIA" in capsys.readouterr().out


def test_analisar_diz_inconclusiva_em_imagem_constante(tmp_path, capsys):
    caminho = tmp_path / "constante.png"
    Image.fromarray(np.full((64, 64), 7, dtype=np.uint8)).save(caminho)
    cli.analisar(str(caminho), 0.5, 10, None)
    assert "INCONCLUSIVA" in capsys.readouterr().out


def test_tabela_usa_o_numero_de_passos_informado(capsys):
    pct = np.array([25.0, 50.0, 75.0, 100.0])
    cli.imprimir_tabela("t", pct, {"x": np.array([0.1, 0.2, 0.3, 0.4])})
    cabecalho = capsys.readouterr().out.splitlines()[2]
    assert "25%" in cabecalho and "100%" in cabecalho and "10%" not in cabecalho


# ---------- Reprodutibilidade dos arquivos ----------
@pytest.mark.parametrize("indice", range(len(CENARIOS_PROJETO)))
# Garante que os arquivos em imagens/exemplos/ são exatamente o que preparar_dados.py gera.
def test_exemplos_em_disco_saem_do_script(indice):
    cenario = CENARIOS_PROJETO[indice]
    arquivo = PASTA_EXEMPLOS / nome_exemplo(cenario)
    if not arquivo.exists() or not (PASTA_KODAK / COVER_EXEMPLO).exists():
        pytest.skip("Rode preparar_dados.py para gerar os exemplos.")
    cover = np.asarray(Image.open(PASTA_KODAK / COVER_EXEMPLO).convert("L"), dtype=np.uint8)
    assert np.array_equal(np.asarray(Image.open(arquivo)), imagem_exemplo(cover, indice, cenario))


def test_povs_contam_todos_os_pixels():
    imagem = gerar_imagem_sintetica(32, 32)
    pares, impares = calcular_povs(imagem)
    assert pares.sum() + impares.sum() == imagem.size
