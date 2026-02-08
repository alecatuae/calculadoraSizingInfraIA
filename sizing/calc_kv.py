"""
Cálculos de KV cache por sessão e total.
"""

from dataclasses import dataclass
from typing import List, Tuple

from .models import ModelSpec


GIB_FACTOR = 2**30


@dataclass
class KVResult:
    """Resultado do cálculo de KV cache."""
    kv_bytes_per_session: float
    kv_gib_per_session: float
    kv_total_gib: float
    kv_total_tib: float
    effective_context_clamped: int
    was_clamped: bool
    warnings: List[str]


def calc_kv_cache(
    model: ModelSpec,
    effective_context: int,
    kv_precision: str,
    concurrency: int
) -> KVResult:
    """
    Calcula KV cache por sessão e total.
    
    Args:
        model: Especificação do modelo
        effective_context: Contexto efetivo solicitado (tokens)
        kv_precision: Precisão do KV cache (fp8, fp16, etc)
        concurrency: Número de sessões simultâneas
    
    Returns:
        KVResult com métricas de KV cache
    """
    warnings = []
    
    # Clamp effective_context ao max_position_embeddings
    effective_context_clamped = effective_context
    was_clamped = False
    
    if effective_context > model.max_position_embeddings:
        effective_context_clamped = model.max_position_embeddings
        was_clamped = True
        warnings.append(
            f"AVISO: effective_context ({effective_context:,}) excede "
            f"max_position_embeddings ({model.max_position_embeddings:,}). "
            f"Ajustado para {effective_context_clamped:,}."
        )
    
    # Obter bytes por elemento
    bytes_per_elem = model.kv_bytes_per_elem(kv_precision)
    
    # Calcular tokens efetivos por camada baseado no attention_pattern
    total_kv_tokens = 0
    
    if model.attention_pattern == "full":
        # Todas as camadas usam contexto completo
        total_kv_tokens = model.num_layers * effective_context_clamped
    
    elif model.attention_pattern == "sliding":
        # Todas as camadas usam sliding_window
        seq_per_layer = min(effective_context_clamped, model.sliding_window)
        total_kv_tokens = model.num_layers * seq_per_layer
    
    elif model.attention_pattern == "hybrid":
        # Algumas camadas full, outras sliding
        full_tokens = model.hybrid_full_layers * effective_context_clamped
        sliding_tokens = model.hybrid_sliding_layers * min(
            effective_context_clamped,
            model.sliding_window
        )
        total_kv_tokens = full_tokens + sliding_tokens
    
    else:
        raise ValueError(f"attention_pattern inválido: {model.attention_pattern}")
    
    # KV cache em bytes por sessão
    # Fórmula: 2 (K e V) × total_tokens × num_kv_heads × head_dim × bytes_per_elem
    kv_bytes_per_session = (
        2 * total_kv_tokens * model.num_key_value_heads * model.head_dim * bytes_per_elem
    )
    
    # Converter para GiB
    kv_gib_per_session = kv_bytes_per_session / GIB_FACTOR
    
    # KV total para a concorrência
    kv_total_gib = kv_gib_per_session * concurrency
    kv_total_tib = kv_total_gib / 1024
    
    # Alertas adicionais
    if kv_precision in ["fp16", "bf16"]:
        warnings.append(
            f"ALERTA: KV precision {kv_precision.upper()} usa 2 bytes/elemento, "
            "dobrando consumo de memória. Considere FP8 (1 byte) para economizar HBM."
        )
    
    if effective_context_clamped > model.max_position_embeddings * 0.9:
        warnings.append(
            f"AVISO: effective_context próximo do máximo ({effective_context_clamped:,} de "
            f"{model.max_position_embeddings:,}). Risco de latência alta (TTFT) e picos."
        )
    
    return KVResult(
        kv_bytes_per_session=kv_bytes_per_session,
        kv_gib_per_session=kv_gib_per_session,
        kv_total_gib=kv_total_gib,
        kv_total_tib=kv_total_tib,
        effective_context_clamped=effective_context_clamped,
        was_clamped=was_clamped,
        warnings=warnings
    )
