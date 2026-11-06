# Projeto 3 — Esteganálise pelo teste do Qui-Quadrado

Detector de esteganografia LSB baseado em Pares de Valores, implementado com NumPy e SciPy. As respostas teóricas e os resultados estão em [`RELATORIO.md`](RELATORIO.md).

## Instalação

Requer Python 3.10 ou superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demonstração rápida

Os comandos abaixo analisam as imagens de exemplo sem alterá-las:

```bash
python main.py --analisar imagens/exemplos/limpa.png
python main.py --analisar imagens/exemplos/sequencial_50.png --grafico demo.png
python main.py --analisar imagens/exemplos/pseudoaleatorio_50.png
```

| Imagem | Resultado esperado |
| :--- | :--- |
| `limpa.png` | sem evidência |
| `sequencial_10.png`, `sequencial_50.png`, `sequencial_100.png` | suspeita; o último ponto suspeito indica onde a mensagem termina |
| `pseudoaleatorio_10.png`, `pseudoaleatorio_50.png` | sem evidência: é a limitação do teste, não um erro do programa |
| `pseudoaleatorio_100.png` | suspeita |

### Cuidado ao demonstrar com uma mensagem própria

O teste supõe bits uniformes, como os de uma mensagem cifrada. Por isso, uma mensagem em **texto puro** ou **muito curta** resulta em "sem evidência", mesmo com a mensagem escondida. Isso é um limite do método, medido na seção 5.5 do relatório, e não um erro do programa:

| Mensagem (imagem `limpa.png`) | Sequencial | Pseudoaleatório |
| :--- | :--- | :--- |
| `dados/mensagem.txt`, cerca de 0,5% da imagem | sem evidência | sem evidência |
| Texto puro, 50% da imagem | sem evidência | sem evidência |
| Bytes aleatórios (simulam cifra), 50% da imagem | **suspeita** | sem evidência |

Para mostrar uma detecção, use `imagens/exemplos/sequencial_50.png` ou uma mensagem de bytes aleatórios:

```bash
python -c "import os; open('cifrada.bin', 'wb').write(os.urandom(24000))"
python main.py --inserir imagens/exemplos/limpa.png --saida stego.png --mensagem cifrada.bin
python main.py --analisar stego.png
```

## Reproduzir o experimento

```bash
python main.py
```

Leva cerca de 1 minuto. Gera as tabelas em `resultados/` e os gráficos em `graficos/`, sempre com os mesmos números, pois todas as sementes são fixas.

Opções: `--covers PASTA`, `--repeticoes N` e `--passos N`.

## Inserir e extrair uma mensagem

```bash
python main.py --inserir imagens/kodak/kodim23.png --saida stego.png --mensagem dados/mensagem.txt --metodo pseudoaleatorio --chave 7
python main.py --extrair stego.png --metodo pseudoaleatorio --chave 7
```

A mensagem de exemplo ocupa menos de 1% da imagem, então o detector não a encontra. Isso é esperado: o relatório mostra que mensagens curtas e espalhadas escapam do teste.

## Regra de decisão

Com `--analisar`, a imagem é **suspeita** se $p \ge 0{,}999$ em algum ponto com pelo menos 10% dos pixels. Os dois valores vêm da calibração descrita no relatório e podem ser alterados com `--limiar` e `--pct-minimo`.

"Sem evidência" não significa que a imagem está limpa. Em imagens coloridas, cada canal é analisado separadamente, além da sequência RGB intercalada.

## Testes

```bash
python -m pytest
```

## Dados

- `imagens/kodak/`: 24 imagens do [Kodak Lossless True Color Image Suite](https://r0k.us/graphics/kodak/), conferidas por SHA-256 (`SHA256SUMS`).
- `imagens/exemplos/`: imagens geradas a partir de `kodim23.png`. Um teste confere se os arquivos correspondem à saída do script.
- `dados/mensagem.txt`: mensagem fictícia de teste.

`python preparar_dados.py` baixa as imagens que faltarem, confere a integridade e gera novamente os exemplos.

## Como estudar o código

O código é comentado para explicar o porquê de cada passo. Uma ordem de leitura que funciona bem:

1. `lsb.py`: como o bit é gravado e lido.
2. `chi_square.py`: o ataque em si. O topo do arquivo traz a ideia e um exemplo numérico.
3. `entrada.py`: por que a leitura da imagem não pode converter as cores.
4. `tests/test_projeto.py`: exemplos curtos de uso de cada função.
5. `experimento.py`: comece por `rodar_experimento`, no fim do arquivo, e siga as chamadas.
6. `main.py` e `graficos_exp.py`: linha de comando e gráficos.

Para ver o efeito na prática, altere um parâmetro em `--analisar` (`--limiar`, `--pct-minimo`, `--passos`) e compare a saída.

## Organização

| Arquivo | Função |
| :--- | :--- |
| `chi_square.py` | PoVs, estatística, varredura e regra de decisão |
| `lsb.py` | Inserção e extração LSB |
| `entrada.py` | Leitura de imagens, preservando os LSBs |
| `experimento.py` | Cenários, calibração e métricas |
| `graficos_exp.py` | Gráficos |
| `preparar_dados.py` | Download das imagens e geração dos exemplos |
| `main.py` | Linha de comando |
| `tests/` | Testes automatizados |
