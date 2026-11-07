<a id="rag-context-study-prepared-execution-deferred"></a>

# Estudo de contexto RAG: preparado, execução adiada

[Comandos e protocolo](../guides/rag-stacking.md) · [Configuração](../../configs/rag-cascade.json)

Marco histórico de implementação, antes da autorização de execução.
O [piloto concluído posteriormente](rag-cascade-pilot.md) substitui seu estado
de execução; os números zero abaixo descrevem somente a preparação.

O usuário pediu explicitamente preparar a implementação sem executar o
experimento porque sua cota de tokens estava baixa. A mudança contém apenas
código, configuração, documentação e testes sintéticos offline.

| Atividade | Estado |
|---|---|
| Recuperação TF-IDF, E5 semântico e híbrida somente no treino | Implementada; execução real adiada |
| Comparação de três/cinco exemplos e quatro ablações de contexto | Preparada; sem novos resultados de qualidade medidos |
| Novas chamadas de classificação GPT | **0** |
| Novos embeddings reais ou downloads de modelos | **0** |
| Novo ajuste de modelos-base ou inferência em dados reais | **0** |
| Novas planilhas de produção teste2 ou avaliações teste1 | **0** |
| Experimentos existentes e modelos congelados | Preservados |

O estudo preparado exclui todas as linhas reservadas do banco RAG. Reutiliza
previsões OOF aninhadas do meta-modelo fixadas por checksum, seleciona recuperação
em pequeno piloto e compara quatro contextos em grupos distintos. Banco,
prompts e seleção nunca usam gabaritos finais. Inferência/avaliação futura
em teste1 fica bloqueada até a mesma configuração congelada ter uma execução
e planilha de teste2 concluídas e inalteradas.

Os controles mantêm cache persistente por resposta/item, limites explícitos
de tentativas, retentativas limitadas, disjuntor e fallback de stacking.
Classificação exige `--execute`; download do codificador, `--download`.
Os padrões só mostram prévias das chamadas.

A suíte offline completa passou em **215 testes em 58.60 segundos**. Uma
regressão posterior de roteamento vazio e verificações finais passaram em
**15 testes RAG**, com embeddings e integração simulados. Contêineres tinham
rede desativada e nenhuma GPU conectada. Testam comportamento da implementação,
não acurácia RAG.

ONNX Runtime e pesos E5 ainda precisam ser preparados quando um estudo real
for autorizado. Nenhum artefato RAG novo foi selecionado ou publicado; a
escolha anterior da cascata sem GPT não muda. Alegações de melhora devem
aguardar o experimento explicitamente adiado.
