# GUIA DE REFERÊNCIA RÁPIDA
# Sistema de Dimensionamento de Inferência LLM

## 🚀 Quick Start

```bash
# Uso básico
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072

# Ver apenas JSON (sem relatório)
python3 sizing.py ... --json-only
```

## 📋 Modelos Disponíveis

| Modelo | Camadas | KV Heads | Max Context | Padrão Atenção |
|--------|---------|----------|-------------|----------------|
| `opt-oss-120b` | 36 | 8 | 131k | hybrid (18 full + 18 sliding) |
| `opt-oss-20b` | 24 | 8 | 131k | hybrid (12 full + 12 sliding) |

## 🖥️ Servidores Disponíveis

| Servidor | GPUs | HBM/GPU | HBM Total | NVLink |
|----------|------|---------|-----------|--------|
| `dgx300` | 8 | 288 GB | 2304 GB (2.3 TB) | 14.4 TB/s |
| `dgx200` | 8 | 141 GB | 1128 GB (1.1 TB) | - |

## 💾 Perfis de Storage

| Perfil | Tipo | IOPS R/W | Throughput | Latência P99 |
|--------|------|----------|------------|--------------|
| `profile_default` | NVMe local | 1M / 800k | 28 / 25 GB/s | 0.15 / 0.20 ms |
| `profile_network_ssd` | Network SSD | 500k / 300k | 12 / 10 GB/s | 2.0 / 3.5 ms |
| `profile_cloud_premium` | Cloud Block | 160k / 120k | 4 / 4 GB/s | 5.0 / 6.0 ms |

## ⚙️ Parâmetros Comuns

```bash
# Precisão KV (impacto direto na memória)
--kv-precision fp8     # 1 byte/elem (recomendado)
--kv-precision fp16    # 2 bytes/elem (dobro da memória)

# Budget de HBM
--kv-budget-ratio 0.70      # 70% para KV cache (padrão)
--runtime-overhead-gib 120  # 120 GiB para modelo + ativações

# Headroom e HA
--peak-headroom-ratio 0.20  # 20% headroom para picos
--ha n+1                    # Alta disponibilidade N+1
--ha none                   # Sem HA (padrão)
```

## 📊 Interpretando Resultados

```json
{
  "results": {
    "kv_per_session_gib": 2.25,      // Memória por sessão
    "kv_total_tib": 2.2,              // Memória total necessária
    "sessions_per_node": 613,         // Capacidade por nó
    "nodes_minimum": 2,               // Nós mínimos (capacidade pura)
    "nodes_with_headroom": 2,         // Nós com headroom de pico
    "nodes_final": 3                  // Nós finais (com HA)
  }
}
```

## 🎯 Cenários Típicos

### Produção com HA
```bash
python3 sizing.py --model opt-oss-120b --server dgx300 \
  --storage profile_default --concurrency 1000 \
  --effective-context 131072 --kv-precision fp8 --ha n+1
# Resultado: 3 nós (2 + N+1)
```

### Desenvolvimento/Testes
```bash
python3 sizing.py --model opt-oss-20b --server dgx200 \
  --storage profile_default --concurrency 100 \
  --effective-context 32768 --kv-precision fp8 --ha none
# Resultado: 1 nó
```

### Alta Precisão (pesquisa)
```bash
python3 sizing.py --model opt-oss-20b --server dgx200 \
  --storage profile_default --concurrency 500 \
  --effective-context 65536 --kv-precision fp16
# Resultado: 2 nós (fp16 dobra memória vs fp8)
```

## 🧮 Fórmulas

### KV Cache por Sessão
```
KV = 2 × seq_length × num_kv_heads × head_dim × bytes_per_elem
```

### Sessões por Nó
```
Budget = (HBM_total × kv_budget_ratio) - runtime_overhead
Sessões = floor(Budget / KV_per_session)
```

### Nós Necessários
```
Nós_mínimos = ceil(concurrency / sessões_por_nó)
Nós_com_headroom = ceil(concurrency × (1 + headroom) / sessões_por_nó)
Nós_finais = Nós_com_headroom + (1 se HA=n+1)
```

## ⚠️ Avisos Comuns

| Aviso | Causa | Solução |
|-------|-------|---------|
| Context excede max_position_embeddings | Context muito grande | Sistema clamp automaticamente |
| fp16 dobra memória | Usando fp16/bf16 | Considere fp8 |
| Prefill pressiona I/O | Context > 128k | Use storage rápido (NVMe) |
| Budget HBM insuficiente | Overhead alto ou budget baixo | Aumente budget ratio ou use servidor maior |

## 🔍 Debugging

```bash
# Ver modelos disponíveis
cat models.json | python3 -m json.tool

# Ver servidores disponíveis
cat servers.json | python3 -m json.tool

# Ver perfis de storage
cat storage.json | python3 -m json.tool

# Executar testes
python3 test_sizing.py
```

## 📞 Suporte

Para adicionar novos modelos, servidores ou perfis de storage, edite os arquivos JSON correspondentes seguindo o formato existente.

---

**Versão:** 1.0  
**Data:** 2026-02-07  
**Python:** 3.8+
