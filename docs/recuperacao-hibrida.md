# Recuperação híbrida de demonstrações — 04/11/2026

## Hipótese e escopo

Melhorar os exemplos enviados ao GPT, mantendo o orçamento de três demonstrações.
O método anterior já recupera exemplos por TF-IDF; esta variante acrescenta
similaridade semântica. É aprendizado em contexto com recuperação de exemplos
rotulados do treino, sem fine-tuning e sem consulta à internet durante a predição.

O estudo usa somente `train_ep1.xlsx`. Teste1 permanece reservado à comparação
com EP1 e teste2 à entrega oficial, conforme o enunciado. O gabarito do teste1
já foi analisado anteriormente: isso deve ser declarado no relatório; não se
reivindica independência de uma futura comparação nesse mesmo teste.

## Implementação

- Encoder: `intfloat/multilingual-e5-small`, revisão
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, saída de 384 dimensões.
- Inferência CPU com ONNX Runtime, quatro threads internas e uma entre operações.
  Pesos FP32 oficiais; não há quantização nem dependência de GPU/PyTorch.
- Tokenização com prefixo `query: ` nos dois lados, por se tratar de similaridade
  simétrica entre respostas. Limite de 512 tokens, média dos estados finais
  mascarada por atenção, normalização L2 e similaridade cosseno.
- O truncamento do encoder afeta apenas a busca. Demonstrações inteiras de até
  1.600 caracteres continuam no prompt; o limite do texto-alvo é o já existente.
- TF-IDF ajustado apenas ao banco de treino elegível, com os mesmos parâmetros
  anteriores. O encoder pré-treinado fica congelado.
- Fusão de rankings por classe: `s(r)=61/(60+rank(r))`, com peso 0,5 para cada
  busca. Empates compartilham posição; o desempate final usa ordem Excel estável.
  A normalização evita misturar diretamente escalas diferentes de cosseno.
- Um exemplo por classe, classes intercaladas. Com mais exemplos, mantém-se
  penalização de redundância, combinando similaridade lexical e semântica.
- `hybrid_length_weight` permite explorar similaridade de comprimento
  `exp(-abs(log(1+len(exemplo))-log(1+len(alvo))))`. **Padrão zero:** comprimento
  não é clareza, e seu benefício não foi demonstrado. Nenhuma regra de estilo
  foi ajustada usando teste1. Os pesos são explícitos na configuração.

Textos normalizados iguais são agrupados; grupos com rótulos conflitantes são
removidos do banco. O próprio grupo do alvo nunca pode ser demonstração.
Alvos chegam à montagem do prompt como `Text`, sem campo de rótulo. A avaliação
local recebe os rótulos separadamente; não são incluídos nos pedidos ao modelo.

## Cache e reprodução

`prepare-retrieval` é a única operação de download do encoder. Os arquivos
ficam no cache do projeto e são fixados por commit. Na classificação, o Hub é
acessado com `local_files_only=True`; não se envia texto ao serviço de embeddings.
Não há execução de código remoto do modelo.

O cache SQLite usa chave com texto limpo, revisão, hashes dos pesos/tokenizador,
pooling, prefixo, limite de tokens, provider e versões das bibliotecas. Não usa
rótulo nem número de linha. Cada lote de embeddings é persistido imediatamente;
uma interrupção preserva o trabalho já concluído. O cache não treina estatísticas
com os alvos e pode ser compartilhado entre partições, pois o encoder é fixo e
cada texto é calculado independentemente. Vetores inválidos provocam erro.

Os hashes do encoder, configuração, banco, código e ambiente também ficam no
manifesto do experimento; mudança de identidade impede retomada incompatível.
O cache de respostas GPT continua separado. Arquivos de resultados antigos não
são reescritos pela nova variante. CPU/bibliotecas diferentes podem causar
pequenas diferenças numéricas; o alias remoto Codex também limita reprodução.

## Protocolo de avaliação

1. Primeiro fold de `StratifiedGroupKFold(5, shuffle=True, seed=42)`. Todo o fold
   fica fora do banco, inclusive registros não amostrados para avaliação.
2. Diagnóstico em até 300 registros, amostrados proporcionalmente por classe.
   Compara vizinho global mais próximo e pureza média dos cinco vizinhos entre
   busca lexical e híbrida. Esses são **proxies de recuperação, não acurácia GPT**;
   a fusão global diagnóstica não usa as cotas por classe do prompt.
3. Piloto pareado em 12 registros desse fold, com prompt original, um exemplo
   por classe, mesmo modelo e mesmo lote de três em ambos os braços. Só varia
   o seletor. Rótulos, métricas, tempos, consumo e exemplos escolhidos são salvos.
4. Prévia das duas variantes antes de qualquer chamada. Teto compartilhado de
   oito processos Codex novos na invocação; se os bytes exigirem mais lotes,
   o estudo recusa iniciar. Sem retentativa automática do adaptador.

O piloto não seleciona configuração de produção: 12 registros não estabelecem
ganho confiável de acurácia, principalmente por classe. Uma próxima comparação
maior deve manter partição, teto de tokens, prompts e plano de análise fixos,
reportar acurácia, macro-F1, recall de c1, tempo e tokens. Ganho no proxy não
implica ganho no GPT. A busca semântica pode recuperar assunto semelhante sem
capturar a avaliação subjetiva; anexos e pedidos originais ausentes continuam
indisponíveis.

## Fontes verificadas

- [Model card oficial E5](https://huggingface.co/intfloat/multilingual-e5-small):
  prefixos, pooling, dimensão e limite de tokens.
- [ONNX Runtime Python](https://onnxruntime.ai/docs/api/python/api_summary.html):
  sessão, provider CPU e entradas NumPy.
- [Downloads do Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/download):
  revisão fixa, cache e modo local. APIs do Tokenizers também conferidas no
  pacote instalado e em sua documentação.
- [Liu et al., 2022](https://aclanthology.org/2022.deelio-1.10/): resultados de
  recuperação de demonstrações em outros benchmarks sustentam investigar a
  hipótese; não são evidência de ganho neste corpus.

Os resultados da execução real serão registrados na seção seguinte.
