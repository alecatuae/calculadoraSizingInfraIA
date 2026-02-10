"""
Schemas de validação para models.json, servers.json e storage.json.

Define estrutura, tipos, campos obrigatórios e constraints de validação.
"""

from typing import Dict, Any, List, Optional

# ============================================================================
# MODEL SCHEMA
# ============================================================================

MODEL_SCHEMA = {
    "required": {
        "name": str,
        "num_layers": int,
        "num_key_value_heads": int,
        "head_dim": int,
        "max_position_embeddings": int,
        "attention_pattern": str,
        "default_kv_precision": str,
    },
    "optional": {
        "hybrid_full_layers": int,
        "hybrid_sliding_layers": int,
        "sliding_window": int,
        "total_params_b": (float, type(None)),
        "active_params_b": (float, type(None)),
        "weights_memory_gib_fp16": (float, type(None)),
        "weights_memory_gib_bf16": (float, type(None)),
        "weights_memory_gib_fp8": (float, type(None)),
        "weights_memory_gib_int8": (float, type(None)),
        "weights_memory_gib_int4": (float, type(None)),
        "default_weights_precision": str,
        "model_artifact_size_gib": (float, type(None)),
        "notes": str,
    },
    "enums": {
        "attention_pattern": ["full", "sliding", "hybrid"],
        "default_kv_precision": ["fp16", "bf16", "fp8", "int8"],
        "default_weights_precision": ["fp16", "bf16", "fp8", "int8", "int4"],
    },
    "constraints": [
        {
            "name": "positive_values",
            "check": lambda m: all([
                m.get("num_layers", 1) > 0,
                m.get("num_key_value_heads", 1) > 0,
                m.get("head_dim", 1) > 0,
                m.get("max_position_embeddings", 1) > 0,
            ]),
            "error": "Architectural parameters (num_layers, num_key_value_heads, head_dim, max_position_embeddings) must be > 0"
        },
        {
            "name": "hybrid_requires_layers",
            "check": lambda m: (
                m.get("attention_pattern") != "hybrid" or 
                (m.get("hybrid_full_layers") is not None and m.get("hybrid_sliding_layers") is not None)
            ),
            "error": "attention_pattern='hybrid' requires hybrid_full_layers and hybrid_sliding_layers"
        },
        {
            "name": "hybrid_sum_equals_total",
            "check": lambda m: (
                m.get("attention_pattern") != "hybrid" or 
                (m.get("hybrid_full_layers", 0) + m.get("hybrid_sliding_layers", 0) == m.get("num_layers", 0))
            ),
            "error": "For hybrid attention: hybrid_full_layers + hybrid_sliding_layers must equal num_layers"
        },
        {
            "name": "sliding_requires_window",
            "check": lambda m: (
                m.get("attention_pattern") not in ["sliding", "hybrid"] or 
                m.get("sliding_window") is not None
            ),
            "error": "attention_pattern='sliding' or 'hybrid' requires sliding_window"
        },
    ]
}

# ============================================================================
# SERVER SCHEMA
# ============================================================================

SERVER_SCHEMA = {
    "required": {
        "name": str,
        "gpus": int,
        "hbm_per_gpu_gb": (int, float),
        "rack_units_u": int,
        "power_kw_max": (int, float),
    },
    "optional": {
        "heat_output_btu_hr_max": ((int, float), type(None)),
        "notes": str,
    },
    "enums": {},
    "constraints": [
        {
            "name": "positive_values",
            "check": lambda s: all([
                s.get("gpus", 1) > 0,
                s.get("hbm_per_gpu_gb", 1) > 0,
                s.get("rack_units_u", 1) > 0,
                s.get("power_kw_max", 1) > 0,
            ]),
            "error": "Server specs (gpus, hbm_per_gpu_gb, rack_units_u, power_kw_max) must be > 0"
        },
    ]
}

# ============================================================================
# STORAGE SCHEMA
# ============================================================================

STORAGE_SCHEMA = {
    "required": {
        "name": str,
        "type": str,
        "capacity_total_tb": (int, float),
        "usable_capacity_tb": (int, float),
        "iops_read_max": int,
        "iops_write_max": int,
        "throughput_read_mbps": (int, float),
        "throughput_write_mbps": (int, float),
        "block_size_kb_read": (int, float),
        "block_size_kb_write": (int, float),
    },
    "optional": {
        "latency_read_ms_p50": ((int, float), type(None)),
        "latency_read_ms_p99": ((int, float), type(None)),
        "latency_write_ms_p50": ((int, float), type(None)),
        "latency_write_ms_p99": ((int, float), type(None)),
        "rack_units_u": int,
        "power_kw": (int, float),
        "notes": str,
    },
    "enums": {},
    "constraints": [
        {
            "name": "positive_values",
            "check": lambda s: all([
                s.get("capacity_total_tb", 1) > 0,
                s.get("usable_capacity_tb", 1) > 0,
                s.get("iops_read_max", 1) > 0,
                s.get("iops_write_max", 1) > 0,
                s.get("throughput_read_mbps", 1) > 0,
                s.get("throughput_write_mbps", 1) > 0,
                s.get("block_size_kb_read", 0.1) > 0,
                s.get("block_size_kb_write", 0.1) > 0,
            ]),
            "error": "Storage specs must be > 0"
        },
        {
            "name": "usable_capacity_le_total",
            "check": lambda s: s.get("usable_capacity_tb", 0) <= s.get("capacity_total_tb", 0),
            "error": "usable_capacity_tb cannot exceed capacity_total_tb"
        },
    ]
}


def get_schema_documentation() -> Dict[str, List[Dict[str, Any]]]:
    """
    Retorna documentação estruturada dos schemas para inclusão no README.
    
    Returns:
        Dict com keys 'models', 'servers', 'storage', cada um contendo
        lista de dicts com: campo, tipo, obrigatório, descrição, unidade, exemplo
    """
    
    models_doc = [
        {"campo": "name", "tipo": "str", "obrigatorio": "Sim", "descricao": "Nome único do modelo", "unidade": "-", "exemplo": "opt-oss-120b"},
        {"campo": "num_layers", "tipo": "int", "obrigatorio": "Sim", "descricao": "Número total de camadas do transformer", "unidade": "layers", "exemplo": "96"},
        {"campo": "num_key_value_heads", "tipo": "int", "obrigatorio": "Sim", "descricao": "Número de cabeças KV (GQA/MQA/MHA)", "unidade": "heads", "exemplo": "8"},
        {"campo": "head_dim", "tipo": "int", "obrigatorio": "Sim", "descricao": "Dimensão de cada cabeça de atenção", "unidade": "dims", "exemplo": "128"},
        {"campo": "max_position_embeddings", "tipo": "int", "obrigatorio": "Sim", "descricao": "Contexto máximo suportado pelo modelo", "unidade": "tokens", "exemplo": "131072"},
        {"campo": "attention_pattern", "tipo": "str", "obrigatorio": "Sim", "descricao": "Padrão de atenção", "unidade": "enum: full|sliding|hybrid", "exemplo": "full"},
        {"campo": "hybrid_full_layers", "tipo": "int", "obrigatorio": "Se hybrid", "descricao": "Número de camadas com atenção full (hybrid)", "unidade": "layers", "exemplo": "48"},
        {"campo": "hybrid_sliding_layers", "tipo": "int", "obrigatorio": "Se hybrid", "descricao": "Número de camadas com atenção sliding (hybrid)", "unidade": "layers", "exemplo": "48"},
        {"campo": "sliding_window", "tipo": "int", "obrigatorio": "Se sliding/hybrid", "descricao": "Tamanho da janela de atenção sliding", "unidade": "tokens", "exemplo": "4096"},
        {"campo": "default_kv_precision", "tipo": "str", "obrigatorio": "Sim", "descricao": "Precisão padrão do KV cache", "unidade": "enum: fp16|bf16|fp8|int8", "exemplo": "fp8"},
        {"campo": "total_params_b", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Parâmetros totais (bilhões)", "unidade": "B", "exemplo": "120.5"},
        {"campo": "active_params_b", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Parâmetros ativos (MoE)", "unidade": "B", "exemplo": "13.0"},
        {"campo": "weights_memory_gib_fp16", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Memória dos pesos em FP16", "unidade": "GiB", "exemplo": "224.4"},
        {"campo": "weights_memory_gib_fp8", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Memória dos pesos em FP8", "unidade": "GiB", "exemplo": "112.2"},
        {"campo": "weights_memory_gib_int8", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Memória dos pesos em INT8", "unidade": "GiB", "exemplo": "112.2"},
        {"campo": "weights_memory_gib_int4", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Memória dos pesos em INT4", "unidade": "GiB", "exemplo": "56.1"},
        {"campo": "default_weights_precision", "tipo": "str", "obrigatorio": "Não", "descricao": "Precisão padrão dos pesos", "unidade": "enum: fp16|bf16|fp8|int8|int4", "exemplo": "fp8"},
        {"campo": "model_artifact_size_gib", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Tamanho do artefato para warmup/storage", "unidade": "GiB", "exemplo": "230.0"},
        {"campo": "notes", "tipo": "str", "obrigatorio": "Não", "descricao": "Notas e observações", "unidade": "-", "exemplo": "Modelo open-source..."},
    ]
    
    servers_doc = [
        {"campo": "name", "tipo": "str", "obrigatorio": "Sim", "descricao": "Nome único do servidor", "unidade": "-", "exemplo": "dgx-b300"},
        {"campo": "gpus", "tipo": "int", "obrigatorio": "Sim", "descricao": "Número de GPUs por nó", "unidade": "count", "exemplo": "8"},
        {"campo": "hbm_per_gpu_gb", "tipo": "float", "obrigatorio": "Sim", "descricao": "Memória HBM por GPU", "unidade": "GB (decimal)", "exemplo": "192.0"},
        {"campo": "rack_units_u", "tipo": "int", "obrigatorio": "Sim", "descricao": "Espaço ocupado em rack", "unidade": "U", "exemplo": "10"},
        {"campo": "power_kw_max", "tipo": "float", "obrigatorio": "Sim", "descricao": "Consumo elétrico máximo", "unidade": "kW", "exemplo": "14.5"},
        {"campo": "heat_output_btu_hr_max", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Dissipação térmica máxima", "unidade": "BTU/hr", "exemplo": "49500.0"},
        {"campo": "notes", "tipo": "str", "obrigatorio": "Não", "descricao": "Notas e observações", "unidade": "-", "exemplo": "NVIDIA DGX B300..."},
    ]
    
    storage_doc = [
        {"campo": "name", "tipo": "str", "obrigatorio": "Sim", "descricao": "Nome único do perfil de storage", "unidade": "-", "exemplo": "profile_default"},
        {"campo": "type", "tipo": "str", "obrigatorio": "Sim", "descricao": "Tipo de storage", "unidade": "-", "exemplo": "nvme_local"},
        {"campo": "capacity_total_tb", "tipo": "float", "obrigatorio": "Sim", "descricao": "Capacidade total bruta", "unidade": "TB", "exemplo": "61.44"},
        {"campo": "usable_capacity_tb", "tipo": "float", "obrigatorio": "Sim", "descricao": "Capacidade utilizável", "unidade": "TB", "exemplo": "56.0"},
        {"campo": "iops_read_max", "tipo": "int", "obrigatorio": "Sim", "descricao": "IOPS máximo de leitura", "unidade": "IOPS", "exemplo": "1000000"},
        {"campo": "iops_write_max", "tipo": "int", "obrigatorio": "Sim", "descricao": "IOPS máximo de escrita", "unidade": "IOPS", "exemplo": "800000"},
        {"campo": "throughput_read_mbps", "tipo": "float", "obrigatorio": "Sim", "descricao": "Throughput máximo de leitura", "unidade": "MB/s (decimal)", "exemplo": "3500.0"},
        {"campo": "throughput_write_mbps", "tipo": "float", "obrigatorio": "Sim", "descricao": "Throughput máximo de escrita", "unidade": "MB/s (decimal)", "exemplo": "3125.0"},
        {"campo": "block_size_kb_read", "tipo": "float", "obrigatorio": "Sim", "descricao": "Tamanho de bloco típico leitura", "unidade": "KB", "exemplo": "3.584"},
        {"campo": "block_size_kb_write", "tipo": "float", "obrigatorio": "Sim", "descricao": "Tamanho de bloco típico escrita", "unidade": "KB", "exemplo": "4.0"},
        {"campo": "latency_read_ms_p50", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Latência leitura (percentil 50)", "unidade": "ms", "exemplo": "0.08"},
        {"campo": "latency_read_ms_p99", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Latência leitura (percentil 99)", "unidade": "ms", "exemplo": "0.15"},
        {"campo": "latency_write_ms_p50", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Latência escrita (percentil 50)", "unidade": "ms", "exemplo": "0.10"},
        {"campo": "latency_write_ms_p99", "tipo": "float|null", "obrigatorio": "Não", "descricao": "Latência escrita (percentil 99)", "unidade": "ms", "exemplo": "0.20"},
        {"campo": "rack_units_u", "tipo": "int", "obrigatorio": "Não", "descricao": "Espaço ocupado em rack", "unidade": "U", "exemplo": "2"},
        {"campo": "power_kw", "tipo": "float", "obrigatorio": "Não", "descricao": "Consumo elétrico", "unidade": "kW", "exemplo": "0.5"},
        {"campo": "notes", "tipo": "str", "obrigatorio": "Não", "descricao": "Notas e observações", "unidade": "-", "exemplo": "Perfil padrão..."},
    ]
    
    return {
        "models": models_doc,
        "servers": servers_doc,
        "storage": storage_doc
    }
