# Prompts da Calculadora de Sizing

Este diretório contém prompts estruturados para desenvolvimento de funcionalidades adicionais da Calculadora de Sizing de Infraestrutura para Inferência.

## 📁 Prompts Disponíveis

### 1. Análise Comparativa de Modelos
**Arquivo**: `analise_comparativa_modelos.md`  
**Objetivo**: Gerar script Python que compara múltiplos relatórios de sizing e identifica o modelo mais eficiente em diferentes dimensões.

**Principais funcionalidades**:
- ✅ Ranking de eficiência de KV cache
- ✅ Comparativo de infraestrutura (nós, VRAM, energia, rack)
- ✅ Análise de custo-benefício (TCO 3 anos)
- ✅ Breakdown de VRAM (modelo fixo vs KV cache vs overhead)
- ✅ Comparação de storage (volumetria, IOPS, throughput)
- ✅ Recomendações executivas por caso de uso
- ✅ Saída em Markdown e JSON

**Casos de uso**:
- Escolher qual modelo LLM adotar para produção
- Avaliar trade-offs entre eficiência de KV e tamanho do modelo
- Estimar TCO para diferentes arquiteturas
- Justificar decisões de infraestrutura para liderança executiva

**Exemplo de uso**:
```bash
python analise_comparativa.py --models "DeepSeek-V3.2,opt-oss-120b" --scenario recommended
```

---

## 🚀 Como Usar os Prompts

1. **Leia o prompt completo**: Cada arquivo `.md` contém especificações detalhadas
2. **Use como entrada para LLM**: Copie o conteúdo e forneça a um modelo de linguagem (GPT-4, Claude, etc.)
3. **Revise o código gerado**: Valide a implementação e adapte conforme necessário
4. **Teste extensivamente**: Execute os testes sugeridos no próprio prompt
5. **Integre ao projeto**: Adicione o script ao repositório e documente no README principal

---

## 🎯 Boas Práticas

### Ao Criar Novos Prompts

1. **Estrutura Clara**:
   - Objetivo (O quê?)
   - Contexto (Por quê?)
   - Requisitos funcionais (Como?)
   - Exemplos de entrada/saída
   - Casos de uso

2. **Especificações Técnicas**:
   - Linguagem e dependências
   - Arquitetura do código (módulos, funções)
   - Validações obrigatórias
   - Formato de saída (JSON schema)

3. **Testes e Validação**:
   - Casos de teste obrigatórios
   - Casos de erro esperados
   - Exemplos de execução

4. **Restrições**:
   - O que NÃO fazer
   - Limitações conhecidas
   - Trade-offs de design

### Ao Implementar a Partir de Prompts

1. ✅ **Valide o prompt**: Certifique-se de que está completo e sem ambiguidades
2. ✅ **Gere incrementalmente**: Não tente implementar tudo de uma vez
3. ✅ **Teste cada módulo**: Valide funções individuais antes de integrar
4. ✅ **Documente divergências**: Se precisar adaptar, documente o motivo
5. ✅ **Atualize o prompt**: Se encontrar melhorias, atualize o prompt original

---

## 📋 Backlog de Prompts Futuros

Ideias para próximos prompts:

### 2. Dashboard Web Interativo
- Interface web para visualizar relatórios de sizing
- Filtros dinâmicos (modelo, servidor, cenário)
- Gráficos comparativos (Chart.js)
- Exportação de relatórios personalizados

### 3. Benchmark de Latência Integrado
- Script para executar benchmarks de TTFT/TPOT
- Integração com vLLM, TensorRT-LLM, TGI
- Correlação entre sizing e performance real
- Validação de premissas da calculadora

### 4. CI/CD para Validação de Modelos
- Pipeline automatizado para testar novos modelos
- Validação de schema do `models.json`
- Sizing automático em múltiplos servidores
- Geração de relatório de compatibilidade

### 5. Estimador de Custo Cloud
- Tradução de sizing on-premise para cloud (AWS, GCP, Azure)
- Comparação de custos entre provedores
- Recomendação de instâncias (p5.48xlarge, etc.)
- TCO on-prem vs cloud

### 6. Otimizador de Configuração
- Algoritmo para encontrar melhor combinação (TP, PP, batch, context)
- Maximizar throughput ou minimizar latência
- Considerar restrições de orçamento
- Sugerir ajustes de `parameters.json`

### 7. Gerador de Relatórios Executivos Personalizados
- Templates customizáveis por organização
- Branded reports (logo, cores)
- Seções opcionais (incluir/excluir métricas)
- Exportação em PDF

### 8. API REST para Sizing
- Endpoint HTTP para sizing via API
- Autenticação e rate limiting
- Cache de resultados
- Documentação OpenAPI/Swagger

---

## 🤝 Contribuindo com Novos Prompts

Se você criar um novo prompt, siga este template:

```markdown
# PROMPT: <Nome Descritivo>

## OBJETIVO
[O que o script/feature deve fazer]

## CONTEXTO
[Por que isso é necessário]

## REQUISITOS FUNCIONAIS
[Especificações detalhadas]

## REQUISITOS TÉCNICOS
[Linguagem, dependências, arquitetura]

## ESTRUTURA DO CÓDIGO
[Módulos, funções principais]

## VALIDAÇÕES OBRIGATÓRIAS
[Testes e casos de erro]

## CASOS DE USO
[Exemplos de uso]

## RESULTADO ESPERADO
[Output esperado]

## RESTRIÇÕES
[O que NÃO fazer]
```

Depois, adicione uma entrada neste README e envie um PR.

---

## 📚 Recursos Adicionais

- **README Principal**: `/README.md`
- **Documentação de Schemas**: `/README_SCHEMAS.md`
- **Arquitetura do Sistema**: `/ARCHITECTURE.md`
- **Guia de Início Rápido**: `/QUICKSTART.md`
- **Schema de Servidores**: `/servers.schema.md`

---

**Última atualização**: 2026-02-13
