# Mapa de Responsabilidades - Projeto Modular

## 📦 Estrutura do Projeto

```
/sizing/                      # Pacote principal
  __init__.py                 # Inicialização do pacote
  cli.py                      # Parse de argumentos CLI
  config_loader.py            # Carregamento de JSON (models, servers, storage)
  models.py                   # Dataclass ModelSpec + validações
  servers.py                  # Dataclass ServerSpec + validações
  storage.py                  # Dataclass StorageProfile + validações
  calc_kv.py                  # Cálculo de KV cache (por sessão e total)
  calc_vram.py                # Cálculo de VRAM (pesos + budget + sessões/nó)
  calc_scenarios.py           # Lógica dos 3 cenários (mínimo/recomendado/ideal)
  calc_physical.py            # Cálculo físico (energia, rack, calor)
  report_full.py              # Geração de relatório completo (texto + JSON)
  report_exec.py              # Geração de resumo executivo (terminal)
  writer.py                   # Escrita de arquivos em ./relatorios

main.py                       # Entrypoint principal (orquestrador)
models.json                   # Especificações de modelos LLM
servers.json                  # Especificações de servidores GPU
storage.json                  # Perfis de storage
README.md                     # Documentação completa
QUICKSTART.md                 # Guia de uso rápido
relatorios/                   # Relatórios gerados (criado em runtime)
```

---

## 🎯 Responsabilidade de Cada Módulo (1 linha)

| Módulo | Responsabilidade Única |
|--------|------------------------|
| `main.py` | Orquestra fluxo completo: CLI → load → calc → report → write → print |
| `cli.py` | Define argparse, retorna CLIConfig com inputs validados |
| `config_loader.py` | Carrega JSON, resolve seleção por nome, valida specs |
| `models.py` | Define ModelSpec, valida attention_pattern, fornece helpers de precisão |
| `servers.py` | Define ServerSpec, valida hardware, calcula HBM total GiB |
| `storage.py` | Define StorageProfile, valida I/O specs (não usado para KV sizing) |
| `calc_kv.py` | Calcula KV cache (bytes/GiB) por sessão e total, clamp context, avisos |
| `calc_vram.py` | Calcula pesos fixos, budget real, sessões/nó, VRAM efetiva, avisos |
| `calc_scenarios.py` | Cria configs de 3 cenários, calcula nodes_final e métricas por cenário |
| `calc_physical.py` | Calcula energia (kW), rack (U) e calor (BTU/hr) por cenário |
| `report_full.py` | Formata relatório completo em texto e JSON (4 seções + alertas) |
| `report_exec.py` | Formata resumo executivo para terminal (tabela de cenários + paths) |
| `writer.py` | Cria ./relatorios, gera nomes com timestamp, escreve txt/json/md |

---

## 🔧 Fluxo de Execução (main.py)

1. **Parse CLI** → `cli.parse_cli_args()` → `CLIConfig`
2. **Load specs** → `ConfigLoader` → `ModelSpec`, `ServerSpec`, `StorageProfile`
3. **Calc KV** → `calc_kv_cache()` → `KVResult`
4. **Calc VRAM** → `calc_vram()` → `VRAMResult` (base)
5. **Calc scenarios** → Para cada cenário:
   - `calc_vram()` com kv_budget_ratio específico
   - `calc_scenario()` → `ScenarioResult`
   - `calc_physical_consumption()` → atualiza ScenarioResult
6. **Generate reports** → `format_full_report()`, `format_json_report()`
7. **Write files** → `ReportWriter.write_text_report()`, `write_json_report()`
8. **Print summary** → `format_exec_summary()` → stdout

---

## ✅ Vantagens da Modularização

### Antes (sizing.py monolítico)
- ❌ 2500+ linhas em arquivo único
- ❌ Lógica misturada (I/O + cálculo + formatação)
- ❌ Difícil testar funções isoladas
- ❌ Difícil adicionar novos cenários ou métricas
- ❌ Imports confusos (tudo no mesmo namespace)

### Depois (projeto modular)
- ✅ Módulos < 200 linhas, responsabilidade clara
- ✅ Separação: I/O (loader, writer) vs Cálculo (calc_*) vs Apresentação (report_*)
- ✅ Fácil testar: cada módulo é uma unidade testável
- ✅ Fácil estender: adicionar novo cenário = editar `calc_scenarios.py`
- ✅ Imports explícitos, namespace limpo

---

## 🚀 Como Estender

### Adicionar Novo Modelo
1. Editar `models.json` com especificações
2. Nenhum código Python precisa mudar!

### Adicionar Novo Servidor
1. Editar `servers.json` com especificações
2. Nenhum código Python precisa mudar!

### Adicionar Nova Métrica (ex: throughput de inferência)
1. Criar `calc_throughput.py` com função pura
2. Chamar de `main.py` após `calc_physical_consumption()`
3. Atualizar `ScenarioResult` em `calc_scenarios.py` com novo campo
4. Atualizar `report_full.py` para exibir nova métrica

### Adicionar Novo Cenário (ex: "Ultra-Conservador")
1. Editar `create_scenario_configs()` em `calc_scenarios.py`
2. Adicionar `"ultra_conservative"` ao dict retornado
3. Atualizar loop em `main.py` e `report_full.py`

### Adicionar Novo Formato de Saída (ex: CSV)
1. Criar método `write_csv_report()` em `writer.py`
2. Chamar de `main.py` após `write_json_report()`

---

## 📚 Convenções de Código

- **Dataclasses** para estruturas de dados (ModelSpec, ServerSpec, etc)
- **Type hints** em todas as funções públicas
- **Docstrings** em funções de cálculo (explicam Args/Returns)
- **Pure functions** para cálculos (sem I/O, sem side effects)
- **Warnings list** retornado de funções de cálculo (nunca print direto)
- **GIB_FACTOR** como constante (2^30) para conversões
- **GB_TO_GIB** como constante (10^9 / 2^30) para conversões GB decimal

---

## 🧪 Testando Módulos Isolados (futuro)

```python
# Exemplo de teste unitário (pytest)
from sizing.calc_kv import calc_kv_cache
from sizing.models import ModelSpec

def test_kv_cache_full_attention():
    model = ModelSpec(
        name="test",
        num_layers=12,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=8192,
        attention_pattern="full"
    )
    result = calc_kv_cache(model, 4096, "fp8", 100)
    
    expected_bytes = 2 * 12 * 4096 * 8 * 64 * 1  # 2 (K+V) × layers × seq × heads × dim × bytes
    assert result.kv_bytes_per_session == expected_bytes
```

---

**Implementado por:** Alexandre  
**Data:** 2026-02-08  
**Versão:** 4.0 (Modular)
